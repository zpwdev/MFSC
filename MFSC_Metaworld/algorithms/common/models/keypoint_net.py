import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import copy
import numpy as np
import cv2
import sys

from .encoder import make_encoder
from .transition_model import make_transition_model
from .utils import norm_mse_loss


epsilon = 1e-8
DEBUG = 0

class KeypointNet3d(nn.Module):
    def __init__(
        self,
        observation_space,
        crop_size,
    ):
        super().__init__()
        frame_stack, self.num_cameras, self.original_size, _, self.image_channels = observation_space['images'].shape
        self.frame_stack = frame_stack
        self.image_size = self.original_size if crop_size is None else crop_size

        self.groups = 1

        self.discount = 0.99

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_dim = 128
        self.state_feature_keys = ['robot_joints']

        self.encoder = make_encoder(obs_shape=(3, 108, 108), hidden_dim=1024, encoder_type='pixel',
                                    encoder_feature_dim=self.feature_dim, num_layers=4, num_filters=64, augmentation=['intensity'], 
                                    aug_prob=1.0, latent_dim=100, num_attn_layers=2, num_heads=1, mask_ratio=0.90, 
                                    jumps=2, patch_size=9, block_size=3, frame_stack=self.frame_stack, output_logits=True)

        self.global_final_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2), nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim))

        self.transition_model = make_transition_model(
            transition_model_type='ensemble_deterministic', encoder_feature_dim=self.feature_dim, action_shape=(4,)
        ).to(self.device)

        print(self.encoder)
    

    def encode(self, obs_all):
        obs = obs_all['images']
        batch_size = obs.shape[0]

        obs = obs.view(batch_size, self.num_cameras, self.frame_stack * self.image_channels,
                             self.image_size, self.image_size)
        obs = obs * 2 - 1
        
        latent = self.encoder(obs, obs_all)
        
        return latent
    
    def get_loss(self, obs_all, actions, rewards, next_obs_all):
        obs = obs_all['images']
        next_obs = next_obs_all['images']
        batch_size = obs.shape[0]

        obs = obs.view(batch_size, self.num_cameras, self.frame_stack * self.image_channels,
                             self.image_size, self.image_size)
        next_obs = next_obs.view(batch_size, self.num_cameras, self.frame_stack * self.image_channels,
                             self.image_size, self.image_size)

        obs = obs * 2 - 1
        next_obs = next_obs * 2 - 1

        # reconstruction_loss
        masked_z_a = self.encoder.mtm_encode(obs, obs_all, mask=True)
        with torch.no_grad():
            target_masked_z_a = self.encoder.mtm_encode(obs, obs_all, mask=False)

        masked_z_a = masked_z_a.flatten(0, 1)
        target_masked_z_a = target_masked_z_a.flatten(0, 1)

        def spr_loss(masked_z_a, target_masked_z_a):
            global_latents = self.global_final_classifier(masked_z_a)
            with torch.no_grad():
                global_targets = target_masked_z_a
            loss = norm_mse_loss(global_latents, global_targets, mean=False).mean()
            return loss

        reconstruction_loss = spr_loss(masked_z_a, target_masked_z_a)

        # bisimulation_loss
        z_a = self.encoder(obs, obs_all)
        with torch.no_grad():
            pred_a = self.transition_model.sample_prediction(
                torch.cat([z_a, actions], dim=1))

        def compute_dis(features_a, features_b):
            # features have been normalized
            similarity_matrix = torch.matmul(features_a, features_b.T)
            dis = 1-similarity_matrix
            return dis

        next_diff = compute_dis(pred_a, pred_a)
        rewards = rewards.unsqueeze(0)
        r_diff = torch.abs(rewards.T - rewards)
        z_diff = compute_dis(z_a, z_a)

        bisimilarity = 0.01 * r_diff + 0.99 * next_diff
        bisimulation_loss = torch.nn.HuberLoss()(z_diff, bisimilarity.detach())

        encoder_loss = reconstruction_loss + bisimulation_loss

        # transition loss
        h = self.encoder(obs, obs_all)
        next_h = self.encoder(next_obs, next_obs_all).unsqueeze(0)
        pred_next_h = self.transition_model(
            torch.cat([h, actions], dim=1))

        cos_sim = F.cosine_similarity(next_h.detach(), pred_next_h, dim=-1)
        transition_loss = 1 - cos_sim.mean()

        # unsup loss
        loss = encoder_loss + transition_loss

        return loss

    def visualize(self, images, keypoints, images_hat, heatmap, u_shift, v_shift):
        pass