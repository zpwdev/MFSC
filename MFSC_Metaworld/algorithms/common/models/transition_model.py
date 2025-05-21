import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeterministicTransitionModel(nn.Module):
    def __init__(self, encoder_feature_dim, action_shape, layer_width):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        return mu

# Add L2 normalization.
class EnsembleOfDeterministicTransitionModels(object):
    def __init__(self, encoder_feature_dim, action_shape, layer_width, ensemble_size=5):
        self.models = [
            DeterministicTransitionModel(encoder_feature_dim, action_shape, layer_width)
            for _ in range(ensemble_size)
        ]
        print("Ensemble of deterministic transition models chosen.")

    def __call__(self, x):
        mu_list = [model.forward(x)[0] for model in self.models]  # Only retrieve mu since sigma is None
        mus = torch.stack(mu_list)
        mus = nn.functional.normalize(mus, dim=-1, p=2)
        return mus

    def sample_prediction(self, x):
        model = random.choice(self.models)
        out = model.sample_prediction(x)
        out = nn.functional.normalize(out, dim=-1, p=2)  # Normalize the output
        return out

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def parameters(self):
        list_of_parameters = [
            list(model.parameters()) for model in self.models
        ]
        parameters = [p for ps in list_of_parameters for p in ps]
        return parameters


_AVAILABLE_TRANSITION_MODELS = {
    '': DeterministicTransitionModel,
    'deterministic': DeterministicTransitionModel,
    'ensemble_deterministic': EnsembleOfDeterministicTransitionModels
}


def make_transition_model(transition_model_type,
                          encoder_feature_dim,
                          action_shape,
                          layer_width=512):
    assert transition_model_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[transition_model_type](
        encoder_feature_dim, action_shape, layer_width)
