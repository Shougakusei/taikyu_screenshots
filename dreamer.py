import torch
import torch.nn as nn
import numpy as np
from d3rlpy.dataset import MDPDataset
import os

from rssm import RSSM, RewardModel, ContinueModel
from encoder import Encoder
from decoder import Decoder
from vae import EdgeEntropyVAE
from actor import Actor
from critic import Critic

from dataset import EdgesDataset, get_seed_list
from torch.utils.data import DataLoader
import torch
import numpy as np

from loss import EdgeDetectionEntropyLoss

from utils import (
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos
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
        verbose=False
    ):
        
        self.config = config.parameters.dreamer
        self.config.in_channels = config.environment.add_screen_count + 1
        
        self.verbose = verbose
        
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool
        
        self.encoder_full = Encoder(config).to(self.device)
        self.encoder_part = Encoder(config).to(self.device)
        self.decoder_full = Decoder(observation_shape, config).to(self.device)
        self.decoder_part = Decoder(observation_shape, config).to(self.device)
        
        self.loss_full = nn.BCELoss()
        self.loss_part = nn.BCELoss()
        
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = Actor(discrete_action_bool, action_size, config).to(self.device)
        self.critic = Critic(config).to(self.device)

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
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.writer = writer
        self.num_total_episode = 0

    def encode(self, obs):
        # Кодируем частичные и полные наблюдения соответствующими энкодерами, затем скрепляем
        return torch.cat([self.encoder_full(obs[:,:,:self.config.in_channels,:,:]), self.encoder_part(obs[:,:,self.config.in_channels:,:,:])],-1)
        
    def decode(self, posteriors, determenistics):
        # Кодируем соответствующее число наблюдений из наших posteriors и determenistics
        reconstructed_observation_full = self.decoder_full(posteriors, determenistics)
        reconstructed_observation_part = self.decoder_part(posteriors, determenistics)
        return reconstructed_observation_full, reconstructed_observation_part
        
    def load_buffer(self, buffer):
        
        # Загружаем буффер в алгоритм
        self.buffer = buffer
        
        # Считаем сердний процент заполненных ячеек на изображении 
        full_sample = self.buffer.sample(10000, 1)['observation'][:,:,:1]
        part_sample = self.buffer.sample(10000, 1)['observation'][:,:,2:]
        self.true_percentage_full = torch.count_nonzero(full_sample) / torch.numel(full_sample)
        self.true_percentage_part = torch.count_nonzero(part_sample) / torch.numel(part_sample)
        
        # Задаем лоссы согласно процентам
        self.loss_full = EdgeDetectionEntropyLoss(self.true_percentage_full)
        self.loss_part = EdgeDetectionEntropyLoss(self.true_percentage_part)
        
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
        
        self.load_buffer(buffer)

        for iteration in range(self.config.train_iterations):
            for collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                self.dynamic_learning(data)
            
                self.num_total_episode += 1

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data, inference_mode=False):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))
        
        if self.verbose:
            print('data.observation.shape: ', data.observation.shape)
        
        data.embedded_observation = self.encode(data.observation)
        
        if self.verbose:
            print('data.embedded_observation.shape: ', data.embedded_observation.shape)
        
        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
#             print(deterministic,  deterministic.shape)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )
        
            if self.verbose:
                print('deterministic.shape: ', deterministic.shape)
        
            self.dynamic_learning_infos.append(
                observation=data.observation,
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior
            
            if self.verbose:
                print('posterior.shape: ', posterior.shape)

        infos = self.dynamic_learning_infos.get_stacked()
        
        self.posteriors_debug, self.determenistics_debug = infos.posteriors, infos.deterministics
        
        self.observations_debug = data.observation
        
        self._model_update(data, infos, inference_mode)
        
        if inference_mode:
            reconstructed_observation_full_1, reconstructed_observation_part_1, reward_dist = self._model_update(data, infos, inference_mode)
            return data.observation[:,:,0,:,:], data.observation[:,:,2,:,:], reconstructed_observation_full_1, reconstructed_observation_part_1, reward_dist
        else:
            return infos.posteriors.detach(), infos.deterministics.detach()
    
    
    def _model_update(self, data, posterior_info, inference_mode):
        
        if self.verbose:
            print('posterior_info.posteriors.shape, posterior_info.deterministics.shape: ', posterior_info.posteriors.shape, posterior_info.deterministics.shape)
        
        
#         reconstructed_observation_dist_full_1 = self.decoder_full(posterior_info.posteriors, posterior_info.deterministics)
#         reconstructed_observation_dist_full_2 = self.decoder_full(posterior_info.posteriors, posterior_info.deterministics)
#         reconstructed_observation_dist_part_1 = self.decoder_part(posterior_info.posteriors, posterior_info.deterministics)
#         reconstructed_observation_dist_part_2 = self.decoder_part(posterior_info.posteriors, posterior_info.deterministics)


        reconstructed_observation_full, reconstructed_observation_part = self.decode(posterior_info.posteriors, posterior_info.deterministics)

        if inference_mode:
        
            reward_dist = self.reward_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
        
            return reconstructed_observation_full_1, reconstructed_observation_part_1, reward_dist

        if inference_mode == False:
            
            reconstructed_observation_full_loss = self.loss_full(reconstructed_observation_full_1, data.observation[:, 1:, 0]) # +\
#             self.loss_full(reconstructed_observation_full_2, data.observation[:, 1:, 1])

            reconstructed_observation_part_loss = self.loss_part(reconstructed_observation_part_1, data.observation[:, 1:, 2]) #+\
#             self.loss_part(reconstructed_observation_part_2, data.observation[:, 1:, 3])
            
            reconstructed_observation_loss = reconstructed_observation_full_loss + reconstructed_observation_part_loss
            
            
            
#             reconstruction_observation_loss =\
#             reconstructed_observation_dist_full_1.log_prob(data.observation[:, 1:, 0]) +\
#             reconstructed_observation_dist_full_2.log_prob(data.observation[:, 1:, 1]) +\
#             reconstructed_observation_dist_part_1.log_prob(data.observation[:, 1:, 2]) +\
#             reconstructed_observation_dist_part_2.log_prob(data.observation[:, 1:, 3]) / 4 / 64 /64

            if self.verbose:
                print('reconstructed_observation_loss: ', reconstructed_observation_loss.mean())

            if self.config.use_continue_flag:
                continue_dist = self.continue_predictor(
                    posterior_info.posteriors, posterior_info.deterministics
                )
                continue_loss = self.continue_criterion(
                    continue_dist.probs, 1 - data.done[:, 1:]
                )

                if self.verbose:
                    print('continue_loss: ', continue_loss.mean())

            reward_dist = self.reward_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            reward_loss = reward_dist.log_prob(data.reward[:, 1:])

            if self.verbose:
                print('reward_loss: ', reward_loss.mean())

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

            if self.verbose:
                print('kl_divergence_loss: ', kl_divergence_loss)

            model_loss = (
                self.config.kl_divergence_scale * kl_divergence_loss
                + reconstructed_observation_loss.mean()
                - reward_loss.mean()
            )
            if self.config.use_continue_flag:
                model_loss += continue_loss.mean()

            if self.verbose:
                print('model_loss: ', model_loss)

            self.model_optimizer.zero_grad()
            model_loss.backward()
            nn.utils.clip_grad_norm_(
                self.model_params,
                self.config.clip_grad,
                norm_type=self.config.grad_norm_type,
            )
            self.model_optimizer.step()

            self.writer.add_scalar(
                                "model_loss", model_loss.item(), self.num_total_episode
                            )
            self.writer.add_scalar(
                                "reconstructed_observation_full_loss", reconstructed_observation_part_loss.mean().item(), self.num_total_episode
                            )
            self.writer.add_scalar(
                                "reconstructed_observation_part_loss", reconstructed_observation_part_loss.mean().item(), self.num_total_episode
                            )
            self.writer.add_scalar(
                                "reward_loss", -reward_loss.mean().item(), self.num_total_episode
                            )
            self.writer.add_scalar(
                                "kl_divergence_loss", self.config.kl_divergence_scale * kl_divergence_loss.item(), self.num_total_episode
            )        
            
            print(reconstructed_observation_full_loss.detach().item(), reconstructed_observation_part_loss.detach().item(), self.config.kl_divergence_scale * kl_divergence_loss.detach().item())
    
    def behavior_learning(self, states, deterministics):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        self._agent_update(self.behavior_learning_infos.get_stacked())

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()
    
    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        for epi in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation = self.encode(
                torch.from_numpy(observation).float().to(self.device)
            )

            score = 0
            score_lst = np.array([])
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action = self.actor(posterior, deterministic).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()

                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action

                next_observation, reward, done, info = env.step(env_action)
                if train:
                    self.buffer.add(
                        observation, buffer_action, reward, next_observation, done
                    )
                score += reward
                embedded_observation = self.encode(
                    torch.from_numpy(next_observation).float().to(self.device)
                )
                observation = next_observation
                if done:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "training score", score, self.num_total_episode
                        )
                    else:
                        score_lst = np.append(score_lst, score)
                    break
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score : ", evaluate_score)
            self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)
            
    def save_params(self, folder_path):
                  
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        torch.save(self.encoder_full, f'{folder_path}/encoder_full.pth')
        torch.save(self.encoder_part, f'{folder_path}/encoder_part.pth')
        torch.save(self.decoder_full, f'{folder_path}/decoder_full.pth')
        torch.save(self.decoder_part, f'{folder_path}/decoder_part.pth')
        torch.save(self.rssm, f'{folder_path}/rssm.pth')
        torch.save(self.reward_predictor, f'{folder_path}/reward_predictor.pth')
        if self.config.use_continue_flag:
            torch.save(self.continue_predictor, f'{folder_path}/continue_predictor.pth')
        torch.save(self.actor, f'{folder_path}/actor.pth')
        torch.save(self.critic, f'{folder_path}/critic.pth')

    def load_params(self, folder_path):
        
        self.encoder_full = torch.load(f'{folder_path}/encoder_full.pth')
        self.encoder_part = torch.load(f'{folder_path}/encoder_part.pth')
        self.decoder_full = torch.load(f'{folder_path}/decoder_full.pth')
        self.decoder_part = torch.load(f'{folder_path}/decoder_part.pth')
        self.rssm = torch.load(f'{folder_path}/rssm.pth')
        self.reward_predictor = torch.load(f'{folder_path}/reward_predictor.pth')
        if self.config.use_continue_flag:
            self.continue_predictor = torch.load(f'{folder_path}/continue_predictor.pth')
        self.actor = torch.load(f'{folder_path}/actor.pth')
        self.critic = torch.load(f'{folder_path}/critic.pth')