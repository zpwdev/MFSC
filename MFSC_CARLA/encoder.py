# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

import copy
import math
from transition_model import make_transition_model
from numpy.core.fromnumeric import shape
from torch.nn import Parameter
from torch.nn import LayerNorm
from kornia.augmentation import (CenterCrop, RandomAffine, RandomCrop,
                                 RandomResizedCrop)
from kornia.filters import GaussianBlur2d
from utils import PositionalEmbedding, maybe_transform, CubeMaskGenerator
from vit_modules import *


class PixelEncoderCarla096(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
             nn.Conv2d(num_filters, num_filters, 3, stride=2),
             nn.Conv2d(num_filters, num_filters, 3, stride=2),
             nn.Conv2d(num_filters, num_filters, 3, stride=2)]
        )
        self.fc = nn.Linear(512, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.mtm = MTM(['intensity'], 1.0, feature_dim, 100, 2, 1, 0.5, 2, 12, 3)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        # print(obs.shape) --> torch.size([1, 9, 84, 420])
        B, F, S, ST = obs.size()
        T = ST // S
        # why must be reshape?
        obs = obs.reshape(B, F, S, T, S)
        obs = obs.permute(0, 3, 1, 2, 4)
        obs = obs.reshape(B * T, F, S, S)

        h = self.forward_conv(obs)

        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        h_norm = h_norm.view(B, T, -1)

        # h1 = torch.mean(h_norm, dim=1)

        cls_token = self.mtm.cls_token
        h_norm = torch.cat([cls_token.expand(B, -1, -1), h_norm], dim=1)

        # add position embedding
        position = self.mtm.position(T+1)
        expand_pos_emb = position.expand(B, -1, -1)  # (B, T, Z)
        h_norm = h_norm + expand_pos_emb
        
        # observation fusion
        for i in range(len(self.mtm.transformer)):
            h_norm = self.mtm.transformer[i](h_norm)

        out = h_norm[:, 0, :]

        return out
    
    def mvm_encode(self, obs, detach=False):
        # print(obs.shape) --> torch.size([1, 9, 84, 420])
        B, F, S, ST = obs.size()
        T = ST // S
        # why must be reshape?
        obs = obs.reshape(B, F, S, T, S)
        obs = obs.permute(0, 3, 1, 2, 4)
        obs = obs.reshape(B * T, F, S, S)

        h = self.forward_conv(obs)

        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        h_norm = h_norm.view(B, T, -1)

        cls_token = self.mtm.cls_token
        h_norm = torch.cat([cls_token.expand(B, -1, -1), h_norm], dim=1)

        # add position embedding
        position = self.mtm.position(T+1)
        expand_pos_emb = position.expand(B, -1, -1)  # (B, T, Z)
        h_norm = h_norm + expand_pos_emb
        
        # observation fusion
        for i in range(len(self.mtm.transformer)):
            h_norm = self.mtm.transformer[i](h_norm)

        out = h_norm[:, 0, :]
        out = nn.functional.normalize(out, dim=-1, p=2)
        
        return out
    
    def mtm_encode(self, obs, mask=True):
        # print(obs.shape) --> torch.size([1, 9, 84, 420])
        B, F, S, ST = obs.size()
        T = ST // S
        obs = obs.reshape(B, F, S, T, S)
        obs = obs.permute(0, 3, 1, 2, 4)
        obs = obs.reshape(B * T, F, S, S)

        if mask == True:
            mask = self.mtm.masker() 
            # print(mask.shape) --> torch.Size([5, 1, 84, 84])
            mask = mask[:, None].expand(mask.size(0), B, *mask.size()[1:]).transpose(0, 1).flatten(0, 1)  # (B*T, ...)
            obs = obs * (1 - mask.float().to(self.device))
            obs = self.mtm.transform(obs, augment=True)

        h = self.forward_conv(obs)
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        h_norm = h_norm.view(B, T, -1)

        # h1 = torch.mean(h_norm, dim=1).unsqueeze(1).expand(B, 6, -1)

        cls_token = self.mtm.cls_token
        h_norm = torch.cat([cls_token.expand(B, -1, -1), h_norm], dim=1)

        position = self.mtm.position(T+1)
        expand_pos_emb = position.expand(B, -1, -1)  # (B, T, Z)
        h_norm = h_norm + expand_pos_emb
        
        for i in range(len(self.mtm.transformer)):
            h_norm = self.mtm.transformer[i](h_norm)
            
        out = h_norm

        return out


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass

class MTM(nn.Module):
    def __init__(self, augmentation, aug_prob, encoder_feature_dim, 
    latent_dim, num_attn_layers, num_heads, mask_ratio, jumps, patch_size, block_size):
        super().__init__()
        # self.mtm = MTM(['intensity'], 1.0, feature_dim, 100, 2, 1, 0.5, 2, 12, 5)
        self.aug_prob = aug_prob
        self.jumps = jumps

        img_size = 84
        input_size = img_size // patch_size


        self.masker = CubeMaskGenerator(
            input_size=input_size, image_size=img_size, clip_size=self.jumps+1, \
                block_size=block_size, mask_ratio=mask_ratio)  # 1 for mask, num_grid=input_size

        self.position = PositionalEmbedding(encoder_feature_dim)
        
        self.state_mask_token = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))

        self.transformer = nn.ModuleList([
            Block(encoder_feature_dim, num_heads, mlp_ratio=2., 
                    qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                    drop_path=0., init_values=0., act_layer=nn.GELU, 
                    norm_layer=nn.LayerNorm, attn_head_dim=None) 
            for _ in range(num_attn_layers)])
        # self.cls_token = torch.randn(1, 1, encoder_feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_feature_dim))

        ''' Data augmentation '''
        self.intensity = Intensity(scale=0.05)
        self.transforms = []
        self.eval_transforms = []
        self.uses_augmentation = True
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1),
                                              (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4),
                                               RandomCrop((84, 84)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.state_mask_token, std=.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(image,
                                        transform,
                                        eval_transform,
                                        p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float(
        ) / 255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None, flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


_AVAILABLE_ENCODERS = {'pixelCarla096': PixelEncoderCarla096,
                       'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=True
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
