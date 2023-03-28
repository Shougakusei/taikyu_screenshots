import torch
import torch.nn as nn

from layers import ConvTranspose2dSame, initialize_weights
from utils import power_calc, create_normal_dist

class Decoder(nn.Module):
    def __init__(self, observation_shape, config, probabilistic=False):
        super(Decoder, self).__init__()
        
        self.config = config.parameters.dreamer.vae
        
        self.config.image_size = config.environment.image_size
        
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        
        self.observation_shape = observation_shape
        
        self.probabilistic = probabilistic
        
        activation = getattr(nn, self.config.activation)()
        
        self.decoder_input = nn.Linear(self.deterministic_size + self.stochastic_size, self.config.num_hiddens*self.config.embedding_image_size**2)
        
        if self.config.num_hiddens != self.config.embedding_dim:
            self._posq_vq_conv = nn.Conv2d(in_channels=self.config.embedding_dim,
                                     out_channels=self.config.num_hiddens,
                                     kernel_size=1, 
                                     stride=1)
            self._posq_vq_conv.apply(initialize_weights)
        
        num_of_convs_transes = power_calc(self.config.image_size,2) - power_calc(self.config.embedding_image_size,2)
        
        conv_trans_layers = [ConvTranspose2dSame(in_channels=self.config.num_hiddens,
                                 out_channels=self.config.num_hiddens // 2,
                                 kernel_size=3,
                                 stride=2), 
                             nn.BatchNorm2d(self.config.num_hiddens // 2),
                             activation]
        
        for i in range(power_calc(self.config.embedding_image_size,2), power_calc(self.config.image_size,2) - 1):
            
            if i != power_calc(self.config.image_size,2) - 2:
                conv_trans_layers.append(ConvTranspose2dSame(in_channels=self.config.num_hiddens//2**(i-1-power_calc(self.config.embedding_image_size,2)+2),
                                 out_channels=self.config.num_hiddens//2**(i-power_calc(self.config.embedding_image_size,2)+2),
                                 kernel_size=3,
                                 stride=2))
                conv_trans_layers.append(nn.BatchNorm2d(self.config.num_hiddens//2**(i-power_calc(self.config.embedding_image_size,2)+2)))
                conv_trans_layers.append(activation)
            else:
                conv_trans_layers.append(ConvTranspose2dSame(in_channels=self.config.num_hiddens//2**(i-1-power_calc(self.config.embedding_image_size,2)+2),
                                 out_channels=self.config.in_channels,
                                 kernel_size=3,
                                 stride=2))
                conv_trans_layers.append(nn.Sigmoid())

        self._conv_transes = nn.Sequential(*conv_trans_layers)
        self._conv_transes.apply(initialize_weights)

    def forward(self, x, y=None):
        
        batch_with_horizon_shape = x.shape[: -len((-1,))]
        if not batch_with_horizon_shape:
            batch_with_horizon_shape = (1,)
        if y is not None:
            x = torch.cat((x, y), -1)
            input_shape = (x.shape[-1],)  #
        x = x.reshape(-1, *input_shape)
        
        x = self.decoder_input(x)
        x = x.view(-1, self.config.num_hiddens, self.config.embedding_image_size, self.config.embedding_image_size)
        
        if self.config.num_hiddens != self.config.embedding_dim:
            x = self._posq_vq_conv(x)
        
        x = self._conv_transes(x)
        
        x = x.reshape(*batch_with_horizon_shape, *[self.config.image_size, self.config.image_size])
        
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        
        if self.probabilistic==True:
            return dist
        elif self.probabilistic==False:
            return x