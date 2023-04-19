import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

import yaml
from attrdict import AttrDict

def find_file(file_name):
    cur_dir = os.getcwd()

    for root, dirs, files in os.walk(cur_dir):
        if file_name in files:
            return os.path.join(root, file_name)

    raise FileNotFoundError(f"File '{file}' not found in subdirectories of {cur_dir}")


def get_base_directory():
    return "/".join(find_file("main.py").split("/")[:-1])

def load_config(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = find_file(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return AttrDict(config)

def power_calc(number, base):
    power = 0
    
    if (number < 1):
        raise ValueError(f'not power of {base}')
        
    while (number != 1):
            if (number % base != 0):
                raise ValueError(f'not power of {base}')
            number = number // base
            power += 1
             
    return power

def pixel_normalization(x):
    return x / 255.0

def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x

def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist

def compute_lambda_values(rewards, values, continues, horizon_length, device, lambda_):
    """
    rewards : (batch_size, time_step, hidden_size)
    values : (batch_size, time_step, hidden_size)
    continue flag will be added
    """
    rewards = rewards[:, 1:]
    continues = continues[:, 1:]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # single step
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * lambda_ * last
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
    return returns

class DynamicInfos:
    def __init__(self, device):
        self.device = device
        self.data = {}

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def get_stacked(self, time_axis=1):
        stacked_data = AttrDict(
            {
                key: torch.stack(self.data[key], dim=time_axis).to(self.device)
                for key in self.data
            }
        )
        self.clear()
        return stacked_data

    def clear(self):
        self.data = {}
        
class PixelNormalization(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return pixel_normalization(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()

        return pixel_normalization(obs)