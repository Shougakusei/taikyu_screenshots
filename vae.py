import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

from utils import create_normal_dist

from abc import abstractmethod

class EdgeDetectionEntropyLoss(nn.Module):
    def __init__(self, true_percentage):
        super(EdgeDetectionEntropyLoss, self).__init__()
        self.true_percentage = true_percentage

    def forward(self, output, target):
        loss = (1 - self.true_percentage) * target * torch.log(output)\
        + self.true_percentage * (1 - target) * torch.log(1 - output)
        return - loss.mean()

class EdgeEntropyVAE(nn.Module):
    def __init__(self, config, true_percentage):
        super(EdgeEntropyVAE, self).__init__()
        
        self.config = config.parameters.dreamer.vae
        
        self.config.image_size = config.environment.image_size
            
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc_mu = nn.Linear(self.config.num_hiddens*self.config.embedding_image_size**2, self.config.embedding_dim)
        self.fc_var = nn.Linear(self.config.num_hiddens*self.config.embedding_image_size**2, self.config.embedding_dim)
        
        self.decoder_input = nn.Linear(self.config.embedding_dim, self.config.num_hiddens*self.config.embedding_image_size**2)
        
        self.loss = EdgeDetectionEntropyLoss(true_percentage)
        
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return [mu, log_var]
    
    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, self.config.num_hiddens, self.config.embedding_image_size, self.config.embedding_image_size)
        z = self.decoder(z)
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        dist = create_normal_dist(mu, std)
        x = dist.rsample()
        return x
    
    def forward(self, x, **kwargs):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]
    
    @abstractmethod
    def loss_f(self, *args, **kwargs):
        output = args[0]
        target = args[1]
        mu = args[2]
        log_var = args[3]
        
        entropy_weight = kwargs['ENTROPY_W']
        
        kld_weight = kwargs['M_N']
        
        recons_loss = self.loss(output, target)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim = 0)
        
        loss = entropy_weight * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
    
    def sample (self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)    
        return samples
    
    def generate(self, x, **kwargs):
        return self.forward(x)[0]