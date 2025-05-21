from typing import Union, Type, Dict, List, Tuple, Optional, Any, Callable
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

import gym
import torch as th
import torch.nn as nn
import numpy as np
import sys

from ..policy_mixins import AugmentPolicyMixin, UnsupPolicyMixin

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..policy_mixins import KeypointPolicyMixin
from ..extractors import KeypointHybridExtractor
from ..models.keypoint_net import KeypointNet3d
from ..utils import preprocess_obs


class AugmentPpoPolicy(AugmentPolicyMixin, ActorCriticPolicy):
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if self.augment:
            obs = self.process_uncropped_obs(obs)
        return super(AugmentPpoPolicy, self).forward(obs)

    def _predict(self, observation: Union[th.Tensor, Dict, Tuple], deterministic: bool = False) -> th.Tensor:
        if self.augment:
            observation = self.process_uncropped_obs(observation)
        return super(AugmentPpoPolicy, self)._predict(observation)


class UnsupPpoPolicy(UnsupPolicyMixin, AugmentPpoPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        augment: bool = False,
        unsup_net_class=None,
        unsup_net_kwargs: dict = {},
        train_jointly: bool = False,
        offset_crop: bool = False,
        first_frame: np.ndarray = None,
    ):
        self.unsup_net_class = unsup_net_class
        self.unsup_net_kwargs = unsup_net_kwargs
        self.train_jointly = train_jointly

        super(UnsupPpoPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            augment=augment,
            offset_crop=offset_crop,
            first_frame=first_frame
        )

    def _get_data(self) -> Dict[str, Any]:
        data = super(UnsupPpoPolicy, self)._get_data()
        data.update(
            train_jointly=self.train_jointly,
            offset_crop=self.offset_crop,
            first_frame=self.first_frame
        )
        return data

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super(UnsupPpoPolicy, self)._build(lr_schedule)
        self.params_exclude_unsup = list(self.parameters())
        self.unsup_net = self.unsup_net_class(self.observation_space, self.crop_size)

        self.optimizer = self.optimizer_class(
            self.params_exclude_unsup + list(self.unsup_net.parameters()),
            lr=0.0002,
            **self.optimizer_kwargs
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        self.features_extractor.eval()
        obs = self.process_uncropped_obs(obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs: Union[th.Tensor, Dict, Tuple]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)

        latent_pi = features
        latent_vf = features
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)

        return latent_pi, latent_vf, latent_sde

    def _predict(self, observation: Union[th.Tensor, Dict, Tuple], deterministic: bool = False) -> th.Tensor:
        self.features_extractor.eval()
        if self.augment:
            observation = self.process_uncropped_obs(observation)
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)


class MultiviewKeypointPpoPolicy(KeypointPolicyMixin, UnsupPpoPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = KeypointHybridExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        augment: bool = False,
        unsup_net_class: KeypointNet3d = None,
        unsup_net_kwargs: dict = {},
        train_jointly: bool = True,
        latent_stack: bool = False,
        offset_crop: bool = False,
        first_frame: np.ndarray = None,
    ):
        unsup_net_kwargs.update(latent_stack=latent_stack)

        super(MultiviewKeypointPpoPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            latent_stack=latent_stack,
            augment=augment,
            unsup_net_class=unsup_net_class,
            unsup_net_kwargs=unsup_net_kwargs,
            train_jointly=train_jointly,
            offset_crop=offset_crop,
            first_frame=first_frame
        )

    def _get_data(self) -> Dict[str, Any]:
        data = super(MultiviewKeypointPpoPolicy, self)._get_data()
        data.update(
            latent_stack=self.latent_stack
        )
        return data

    def _get_latent(self, obs: Union[th.Tensor, Dict, Tuple]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = self.process_multiview_obs(obs)
        return super(MultiviewKeypointPpoPolicy, self)._get_latent(obs)
    
    def _get_loss(self, obs, actions, rewards, next_obs):
        obs = self.process_multiview_obs(obs)
        next_obs = self.process_multiview_obs(next_obs)
        loss_dict = self.extract_loss(obs, actions, rewards, next_obs)
        return loss_dict

    def extract_features(self, obs):
        return super(MultiviewKeypointPpoPolicy, self).extract_features(obs, not self.train_jointly)

    def evaluate_actions(self, obs, actions, rewards, next_obs):
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        unsup_loss = self._get_loss(obs, actions, rewards, next_obs)

        return values, log_prob, distribution.entropy(), unsup_loss

    def visualize(self, obs):
        pass