import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from transition_model import make_transition_model
import utils
from utils import norm_mse_loss, mlp
from encoder import make_encoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        # self.encoder = make_encoder(
        #     encoder_type, obs_shape, encoder_feature_dim, num_layers,
        #     num_filters, output_logits=True
        # )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        # self.encoder = make_encoder(
        #     encoder_type, obs_shape, encoder_feature_dim, num_layers,
        #     num_filters, output_logits=True
        # )

        self.Q1 = QFunction(
            encoder_feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            encoder_feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return
        
        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class SimSR(nn.Module):
    """
    SimSR
    """

    def __init__(self, obs_shape, z_dim, batch_size, encoder, encoder_target, output_type="continuous"):
        super(SimSR, self).__init__()
        self.batch_size = batch_size

        self.encoder = encoder

        self.encoder_target = encoder_target

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out


class SimSRSacAgent(object):
    """SimSR representation learning with SAC."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128,
        transition_model_type='ensemble'
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
        ).to(device)

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.encoder_target = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        # copy encoders
        self.encoder_target.load_state_dict(copy.deepcopy(self.encoder.state_dict()))

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        decoder_weight_lambda = 0.0000001
        self.transition_optimizer = torch.optim.Adam(
            self.transition_model.parameters(), lr=encoder_lr, weight_decay=decoder_weight_lambda
        )

        if self.encoder_type == 'pixelCarla096':
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.SimSR = SimSR(obs_shape, encoder_feature_dim,
                               self.curl_latent_dim, self.encoder, self.encoder_target, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.encoder.parameters(), lr=encoder_lr
            )
        
        print(self.encoder)

        self.train()
        self.critic_target.train()
        if transition_model_type == 'ensemble':
            for i in range(len(self.transition_model.models)):
                self.transition_model.models[i].train()
        else:
            self.transition_model.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixelCarla096':
            self.SimSR.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            obs = obs.unsqueeze(0)
            obs = self.encoder(obs)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        # if obs.shape[-1] != self.image_size:
        #     obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            obs = obs.unsqueeze(0)
            obs = self.encoder(obs)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            next_obs_ = self.encoder(next_obs)
            _, policy_action, log_pi, _ = self.actor(next_obs_)

            next_obs_ = self.encoder_target(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs_, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        obs = self.encoder(obs)
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        # torch.nn.utils.clip_grad_norm(parameters=self.critic.parameters(), max_norm=0.5, norm_type=2)
        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        with torch.no_grad():
            obs = self.encoder(obs)
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        # torch.nn.utils.clip_grad_norm(parameters=self.actor.parameters(), max_norm=0.5, norm_type=2)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_encoder(self, obs, next_obs, reward, action, L, step):
        
        # reconstruction_loss
        masked_z_a = self.encoder.mtm_encode(obs, mask=True)
        with torch.no_grad():
            target_masked_z_a = self.encoder_target.mtm_encode(obs, mask=False)

        masked_z_a = masked_z_a.flatten(0, 1)
        target_masked_z_a = target_masked_z_a.flatten(0, 1)

        def spr_loss(masked_z_a, target_masked_z_a):
            global_latents = masked_z_a
            with torch.no_grad():
                global_targets = target_masked_z_a
            loss = norm_mse_loss(global_latents, global_targets, mean=False).mean()
            return loss

        reconstruction_loss = spr_loss(masked_z_a, target_masked_z_a)

        # bisimulation_loss
        z_a = self.encoder.mvm_encode(obs)

        with torch.no_grad():
            z_a_tar = self.encoder_target.mvm_encode(obs)
            pred_a = self.transition_model.sample_prediction(
                torch.cat([z_a_tar, action], dim=1))

        def compute_dis(features_a, features_b):
            # features have been normalized
            similarity_matrix = torch.matmul(features_a, features_b.T)
            dis = 1-similarity_matrix
            return dis

        r_diff = torch.abs(reward.T - reward)
        next_diff = compute_dis(pred_a, pred_a)
        z_diff = compute_dis(z_a, z_a)
        bisimilarity = 0.01 * r_diff + self.discount * next_diff
        bisimulation_loss = torch.nn.HuberLoss()(z_diff, bisimilarity.detach())

        loss = bisimulation_loss + reconstruction_loss

        if step % self.log_interval == 0:
            L.log('train/SimSR_loss', loss, step)
            L.log('train/bisimilarity', bisimilarity.mean().item(), step)
            L.log('train/r_dist', r_diff.mean().item(), step)
        return loss

    def update_transition_model(self, obs, action, next_obs, L, step):
        h = self.encoder(obs)
        pred_next_latent_mu = self.transition_model(
            torch.cat([h, action], dim=1))
        next_h = self.encoder(next_obs)
        next_h = nn.functional.normalize(next_h, dim=-1, p=2)

        cos_sim = F.cosine_similarity(next_h.detach(), pred_next_latent_mu, dim=-1)
        loss = 1 - cos_sim.mean()

        L.log('train_ae/transition_loss', loss, step)
        return loss

    def update(self, replay_buffer, L, step):

        # obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if self.encoder_type == 'pixelCarla096':
            transition_loss = self.update_transition_model(
                obs, action, next_obs, L, step)
            encoder_loss = self.update_encoder(
                obs, next_obs, reward, action, L, step)
            total_loss = encoder_loss + transition_loss
            self.encoder_optimizer.zero_grad()
            self.transition_optimizer.zero_grad()
            total_loss.backward()
            self.encoder_optimizer.step()
            self.transition_optimizer.step()

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.encoder, self.encoder_target,
                self.encoder_tau
            )

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_SimSR(self, model_dir, step):
        torch.save(
            self.encoder.state_dict(), '%s/SimSR_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

    def load_SimSR(self, model_dir, step):
        print(model_dir, step)
        self.critic.encoder.load_state_dict(
            torch.load('%s/SimSR_%s.pt' % (model_dir, step))
        )
        self.actor.encoder.load_state_dict(
            torch.load('%s/SimSR_%s.pt' % (model_dir, step))
        )
        print('successfully load')
