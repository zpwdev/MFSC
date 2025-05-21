import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

from .vit_modules import *
from .masking_generator import RandomMaskingGenerator, CubeMaskGenerator

import copy
import math
from .transition_model import make_transition_model
from numpy.core.fromnumeric import shape
from torch.nn import Parameter
from torch.nn import LayerNorm
from kornia.augmentation import (CenterCrop, RandomAffine, RandomCrop,
                                 RandomResizedCrop)
from kornia.filters import GaussianBlur2d
from .utils import PositionalEmbedding, InverseSquareRootSchedule, AnneallingSchedule, maybe_transform
import torchvision.transforms._transforms_video as v_transform

import sys

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# for 108 x 108 inputs
OUT_DIM_108 = {2: 51, 4: 47, 6: 43}
# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

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

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, hidden_dim, encoder_type,
                feature_dim, num_layers, num_filters, augmentation, 
                aug_prob, latent_dim, num_attn_layers, num_heads, mask_ratio, 
                jumps, patch_size, block_size, frame_stack, output_logits=True
                 ):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        self.convs = nn.ModuleList([nn.Conv2d(frame_stack * 3, num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_108[num_layers] if obs_shape[-1] == 108 else OUT_DIM[
            num_layers]
        
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.linear = nn.Linear(feature_dim + frame_stack, feature_dim)
        
        # includes two functional components
        # image augmentation and mask preprocessing, as well as the Transformer module.
        self.mtm = MTM(augmentation, aug_prob, feature_dim, 
                latent_dim, num_attn_layers, num_heads, 
                mask_ratio, jumps, patch_size, block_size)
        
        # Add supplementary state information.
        self.state_feature_keys = ['robot_joints']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.apply(weight_init)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs, flatten=True):
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = conv.view(conv.size(0), -1) if flatten else conv        
        return h

    def forward(self, obs, obs_all, detach=False):
        B, T, F, S, S = obs.size()

        obs = obs.view(B * T, F, S, S)

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

        if self.state_feature_keys:
            state_feature = [v.flatten(1) for k, v in obs_all.items() if k in self.state_feature_keys]
            out = th.cat([out] + state_feature, 1)
        out = self.linear(out)
        out = nn.functional.normalize(out, dim=-1, p=2)
        
        return out
    
    # for reconstruction
    def mtm_encode(self, obs, obs_all, mask=True):
        B, T, F, S, S = obs.size()
        obs = obs.view(B * T, F, S, S)

        if mask == True:
            mask = self.mtm.masker() 
            mask = mask[:, None].expand(mask.size(0), B, *mask.size()[1:]).transpose(0, 1).flatten(0, 1)  # (B*T, ...)
            obs = obs * (1 - mask.float().to(self.device))
            obs = self.mtm.transform(obs, augment=True)

        h = self.forward_conv(obs)
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        h_norm = h_norm.view(B, T, -1)

        cls_token = self.mtm.cls_token
        h_norm = torch.cat([cls_token.expand(B, -1, -1), h_norm], dim=1)

        position = self.mtm.position(T+1)
        expand_pos_emb = position.expand(B, -1, -1)  # (B, T, Z)
        h_norm = h_norm + expand_pos_emb
        
        for i in range(len(self.mtm.transformer)):
            h_norm = self.mtm.transformer[i](h_norm)
        
        out = h_norm

        return out


class MTM(nn.Module):
    def __init__(self, augmentation, aug_prob, encoder_feature_dim, 
    latent_dim, num_attn_layers, num_heads, mask_ratio, jumps, patch_size, block_size):
        super().__init__()
        self.aug_prob = aug_prob
        self.jumps = jumps

        img_size = 108
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

def make_encoder(obs_shape, hidden_dim, encoder_type,
                encoder_feature_dim, num_layers, num_filters, augmentation, 
                aug_prob, latent_dim, num_attn_layers, num_heads, mask_ratio, 
                jumps, patch_size, block_size, frame_stack, output_logits=True
                 ):
    
    return PixelEncoder(obs_shape, hidden_dim, encoder_type,
                        encoder_feature_dim, num_layers, num_filters, augmentation, 
                        aug_prob, latent_dim, num_attn_layers, num_heads, mask_ratio, 
                        jumps, patch_size, block_size, frame_stack, output_logits=True)