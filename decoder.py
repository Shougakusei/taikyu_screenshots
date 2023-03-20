import torch
import torch.nn as nn

from layers import ConvTranspose2dSame, initialize_weights
from utils import power_calc

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        self.config = config.parameters.dreamer.vae
        
        self.config.image_size = config.environment.image_size

        activation = getattr(nn, self.config.activation)()
        
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

    def forward(self, x):
        
        if self.config.num_hiddens != self.config.embedding_dim:
            x = self._posq_vq_conv(x)
        
        x = self._conv_transes(x)
        
        return x