import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple
import numpy as np

class KeypointHybridExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 128,
    ):
        super(KeypointHybridExtractor, self).__init__(
            observation_space,
            features_dim=features_dim
        )

    def forward(self, observations: [th.Tensor, Dict, Tuple]) -> th.Tensor:
        pass