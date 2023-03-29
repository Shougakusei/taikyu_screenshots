import torch
import torch.nn as nn
from utils import create_normal_dist
from layers import build_network


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.critic
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        
        batch_with_horizon_shape = posterior.shape[: -len((-1,))]
        x = torch.cat((posterior, deterministic), -1)
        input_shape = (x.shape[-1],)
        x = x.reshape(-1, *(-1,))
        
        x = self.network(x)
        
        x = x.reshape(*batch_with_horizon_shape, *(1,))
        
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist