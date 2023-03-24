import torch
import torch.nn as nn
import numpy as np
from d3rlpy.dataset import MDPDataset

from rssm import RSSM, RewardModel, ContinueModel
from encoder import Encoder
from decoder import Decoder
from vae import EdgeEntropyVAE
# from actor import Actor
# from critic import Critic

from dataset import EdgesDataset, get_seed_list
from utils import load_config
from torch.utils.data import DataLoader
import torch
import numpy as np

from utils import (
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos,
)
from buffer import ReplayBuffer


class Dreamer:
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
    ):
        
        self.config = config.parameters.dreamer
        
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool
        
        self.encoder_full = Encoder(config).to(self.device)
        self.encoder_part = Encoder(config).to(self.device)
        self.decoder_full = Decoder(config).to(self.device)
        self.decoder_part = Decoder(config).to(self.device)
        
#         self.encoder = Encoder(observation_shape, config).to(self.device)
#         self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
#         self.actor = Actor(discrete_action_bool, action_size, config).to(self.device)
#         self.critic = Critic(config).to(self.device)

        self.buffer = ReplayBuffer(observation_shape, action_size, config)

        

        # optimizer
        self.model_params = (
            list(self.encoder_full.parameters())
            + list(self.encoder_part.parameters())
            + list(self.decoder_full.parameters())
            + list(self.decoder_part.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
#         self.actor_optimizer = torch.optim.Adam(
#             self.actor.parameters(), lr=self.config.actor_learning_rate
#         )
#         self.critic_optimizer = torch.optim.Adam(
#             self.critic.parameters(), lr=self.config.critic_learning_rate
#         )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.writer = writer
        self.num_total_episode = 0
    
    def train(self, env):
        if len(self.buffer) < 1:
            self.environment_interaction(env, self.config.seed_episodes)

        for iteration in range(self.config.train_iterations):
            for collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                posteriors, deterministics = self.dynamic_learning(data)
                self.behavior_learning(posteriors, deterministics)

            self.environment_interaction(env, self.config.num_interaction_episodes)
            self.evaluate(env)
            
    def train_offline(self, buffer):
        
        self.buffer = buffer

        for iteration in range(self.config.train_iterations):
            for collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                self.dynamic_learning(data)

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))
        
        print(data.observation)
        
        data.embedded_observation = torch.cat([self.encoder_full(data.observation[:,:,0,:,:]),self.encoder_full(data.observation[:,:,1,:,:]),
                                               self.encoder_part(data.observation[:,:,2,:,:]),self.encoder_part(data.observation[:,:,3,:,:])],-1)

        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, data, posterior_info):
        
        reconstructed_observation_dist = torch.cat([
            self.decoder_full(posterior_info.posteriors, posterior_info.deterministics),
            self.decoder_full(posterior_info.posteriors, posterior_info.deterministics),
            self.decoder_part(posterior_info.posteriors, posterior_info.deterministics),
            self.decoder_part(posterior_info.posteriors, posterior_info.deterministics),
                                                   ],2)
        
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()