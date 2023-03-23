import torch
import torch.nn as nn

from layers import Conv2dSame, initialize_weights
from utils import power_calc

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.config = config.parameters.dreamer.vae
        
        self.config.image_size = config.environment.image_size

        activation = getattr(nn, self.config.activation)()
        
        num_of_convs = power_calc(self.config.image_size,2) - power_calc(self.config.embedding_image_size,2)
        
        conv_layers = [Conv2dSame(in_channels=self.config.in_channels,
                                 out_channels=self.config.num_hiddens // 2**(num_of_convs - 1),
                                 kernel_size=3,
                                 stride=2), 
                       nn.BatchNorm2d(self.config.num_hiddens // 2**(num_of_convs - 1)),
                       activation]
        
        for i in reversed(range(1, num_of_convs)):
            conv_layers.append(Conv2dSame(in_channels=self.config.num_hiddens//2**(i),
                                 out_channels=self.config.num_hiddens//2**(i-1),
                                 kernel_size=3,
                                 stride=2))
            conv_layers.append(nn.BatchNorm2d(self.config.num_hiddens//2**(i-1)))
            conv_layers.append(activation)
        
        self._convs = nn.Sequential(*conv_layers)
        self._convs.apply(initialize_weights)
        
        if self.config.num_hiddens != self.config.embedding_dim:
            self._pre_vq_conv = nn.Conv2d(in_channels=self.config.num_hiddens, 
                                          out_channels=self.config.embedding_dim,
                                          kernel_size=1, 
                                          stride=1)
            self._pre_vq_conv.apply(initialize_weights)

    def forward(self, x):
        
        batch_with_horizon_shape = x.shape[: -len([self.config.image_size, self.config.image_size])]
        if not batch_with_horizon_shape:
            batch_with_horizon_shape = (1,)
        x = x.reshape(-1, *[self.config.image_size, self.config.image_size])
        x = x.unsqueeze(1)
        
        x = self._convs(x)
        
        if self.config.num_hiddens != self.config.embedding_dim:
            x = self._pre_vq_conv(x)
        
        x = x.reshape(*batch_with_horizon_shape, *(-1,))
        
        return x