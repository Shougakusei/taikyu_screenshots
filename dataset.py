import os
import re

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import random
import torchvision.utils as vutils
import argparse
import math
import cv2
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from functools import reduce

from torchsummary import summary

from abc import abstractmethod

def extract_metadata(fpath):
    
    with open(fpath, 'r') as file:
        info = file.read().rstrip()
        
    info_splitted = info.split('_')
    
    timestep = int(info_splitted[0])
    reward = float(info_splitted[1])
    player_x = float(info_splitted[2])
    player_y = float(info_splitted[3])
    on_platform = bool(info_splitted[4]) 
    djump = bool(info_splitted[5])
    
#     movements = info_splited[4].split(';')[:-1]
    
    walk_distance = jump_height = jump_true = 0
    
#     Рассчитываем действие из последовательности мувментов
#     for i in range(len(movements)):
        
#         movement_list = [int(x) for x in list(movements[i])]
        
#         if movement_list[0] == 1:
#             walk_distance -= 1
            
#         if movement_list[1] == 1:
#             walk_distance += 1
        
#         if movement_list[2] == 1:
#             jump_true = 1
        
#         # Если jumpt_true == 0, то не делаем ничего
#         if (movement_list[3] == 0) and (jump_true == 1):
#             jump_height += 1
#         elif (movement_list[3] == 1) and (jump_true == 1):
#             jump_height += 1
#             jump_true = 0
        
    return timestep, reward, player_x, player_y, on_platform, djump

def take_screen_part(img, player_x, player_y, width, height, pad):
    
    img = np.pad(img,pad)
    return img[pad+player_y-height:pad+player_y+height,pad+player_x-width:pad+player_x+width]

def fullscreen_transform(img, dim, pad = 32, zero_screenshot=None, aperture_size=7):
    
    # Обрезаем рамку со статичными элементами
    img = img[pad:-pad, pad:-pad]
    # Ресайз скрина до размера наблюдения
    img = cv2.resize(img, dim)
    # Edge detection
    img = cv2.Canny(img,150,250, apertureSize=aperture_size)
    # Если есть zero_screenshot вычитаем его из наблюдения
    if zero_screenshot is not None:
            img = (img != zero_screenshot) * img
    return img

def partscreen_transform(img, dim, zero_screenshot=None, aperture_size=7):
    
    # Ресайз скрина до размера наблюдения
    img = cv2.resize(img, dim)
    # Edge detection
    img = cv2.Canny(img,150,250, apertureSize=aperture_size)
    # Если есть zero_screenshot вычитаем его из наблюдения
    if zero_screenshot is not None:
            img = (img != zero_screenshot) * img
    return img

def transform_img(img, dim, part_size=None, player_x=None, player_y=None, zero_screenshot=None, aperture_size=7):
    
    if part_size==None:
        img = fullscreen_transform(img, dim, zero_screenshot=zero_screenshot, aperture_size=aperture_size)
    else:
        img = take_screen_part(img, player_x, player_y, part_size, part_size, part_size)
        img = partscreen_transform(img, dim, zero_screenshot=zero_screenshot, aperture_size=aperture_size)
    return img

def transform_multiproc(img, zero_screenshot, zero_screenshot_part, config):
    """Возвращает тупл вида (полный скриншот, частичный скриншот), если нужны оба или просто отдельный скриншот"""
    if config.environment.full_screen_obs and config.environment.part_screen_obs:
        return transform_img(img, 
                   dim=[config.environment.image_size, config.environment.image_size], 
                   zero_screenshot=zero_screenshot, 
                   aperture_size=config.parameters.edges_dataset.aperture_size),\
                transform_img(img, 
                    dim=[config.environment.image_size, config.environment.image_size],
                    zero_screenshot=zero_screenshot_part, 
                    part_size=config.parameters.edges_dataset.part_size,
                    player_x=config.parameters.edges_dataset.zero_screenshot_player_x,
                    player_y=config.parameters.edges_dataset.zero_screenshot_player_y,
                    aperture_size=config.parameters.edges_dataset.aperture_size)
    elif config.environment.full_screen_obs:
        return transform_img(img, 
                   dim=[config.environment.image_size, config.environment.image_size], 
                   zero_screenshot=zero_screenshot, 
                   aperture_size=config.parameters.edges_dataset.aperture_size)
    elif config.environment.part_screen_obs:
        return transform_img(img, 
                    dim=[config.environment.image_size, config.environment.image_size],
                    zero_screenshot=zero_screenshot_part, 
                    part_size=config.parameters.edges_dataset.part_size,
                    player_x=config.parameters.edges_dataset.zero_screenshot_player_x,
                    player_y=config.parameters.edges_dataset.zero_screenshot_player_y,
                    aperture_size=config.parameters.edges_dataset.aperture_size)

def zero_screenshot_load(zero_screenshot_path, image_size, pad, aperture_size):
    '''Загружаем и трансформируем нулевой скриншот, приводим к формату наблюдения'''
    zero_screenshot = cv2.imread(zero_screenshot_path, 0)
    zero_screenshot = fullscreen_transform(zero_screenshot, dim=[image_size,image_size], pad=pad, aperture_size=aperture_size)
    return zero_screenshot

def zero_screenshot_part_load(zero_screenshot_path, image_size, part_size, player_x, player_y, pad, aperture_size):
    '''Загружаем и трансформируем частичный нулевой скриншот, приводим к формату наблюдения'''
    zero_screenshot = cv2.imread(zero_screenshot_path, 0)
    zero_screenshot = take_screen_part(zero_screenshot, player_x, player_y, part_size, part_size, part_size)
    zero_screenshot = partscreen_transform(zero_screenshot, dim=[image_size,image_size], aperture_size=aperture_size)
    return zero_screenshot


def get_seed_list(path):
    return list(map(lambda x: int(x[len('screenshots0000_'):]),os.listdir(path)))

def load_image_obs(image_path, zero_screenshot, zero_screenshot_part, config, multiprocessing=False):
    
    # Добавляем в список пути к доп скриншотам начиная с самого раннего
    image_pathes= []
    image_path_add = re.sub(r'/screenshots/', r'/screenshots_add/', image_path)
    for i in range(config.environment.add_screen_count):
        image_pathes.append(image_path_add[:-4] + f'_{i}' + image_path_add[-4:]) 
        
    # Добавляем основной скриншот
    image_pathes.append(
        image_path)
    
    # Путь к метаданным
    metadata_path = image_path_add[:-3] + 'txt'
    timestep, reward, player_x, player_y, on_platform, djump = extract_metadata(metadata_path)
    
    # Рассчитываем окончание эпизода
    if timestep < config.environment.timestep_max:
        timestep_next = timestep + config.environment.timestep_stride
        terminal = False
    else:
        terminal = True

        
    # флаг 0 = читаем черно-белое изображение
    
    if multiprocessing:
        imgs = Parallel(n_jobs = (config.environment.add_screen_count + 1) * 2)(delayed(cv2.imread)(image_path, 0) for image_path in image_pathes)
    else:
        imgs = [cv2.imread(image_path, 0) for image_path in image_pathes]
    
    if multiprocessing:
        
        if config.environment.full_screen_obs and config.environment.part_screen_obs:
            # Получаем параллельными вычислениями лист туплов вида (полный скрин, частичный скрин)
            observation_img = Parallel(n_jobs = (config.environment.add_screen_count + 1) * 2 )(delayed(transform_multiproc)(img, zero_screenshot, zero_screenshot_part, config) for img in imgs)
            # Сливаем в одномерный лист
            observation_img = reduce(lambda x,y :x+y ,imgs)
        else:
            # Получаем параллельными вычислениями лист скриншотов
            observation_img = Parallel(n_jobs = (config.environment.add_screen_count + 1))(delayed(transform_multiproc)(img, zero_screenshot, zero_screenshot_part, config) for img in imgs)
        
    else:
        # Трансформируем и получаем готовые полные изображения
        if config.environment.full_screen_obs:
            imgs_full = [transform_img(img, 
                           dim=[config.environment.image_size, config.environment.image_size], 
                           zero_screenshot=zero_screenshot, 
                           aperture_size=config.parameters.edges_dataset.aperture_size)
                        for img in imgs]

        # Трансформируем и получаем частичные изображения
        if config.environment.part_screen_obs:
            imgs_part = [transform_img(img, 
                            dim=[config.environment.image_size, config.environment.image_size],
                            zero_screenshot=zero_screenshot_part, 
                            part_size=config.parameters.edges_dataset.part_size,
                            player_x=config.parameters.edges_dataset.zero_screenshot_player_x,
                            player_y=config.parameters.edges_dataset.zero_screenshot_player_y,
                            aperture_size=config.parameters.edges_dataset.aperture_size)
                        for img in imgs]
            
        if config.environment.full_screen_obs and config.environment.part_screen_obs:
            observation_img = np.stack([*imgs_full, *imgs_part])
        elif config.environment.full_screen_obs:
            observation_img = np.stack([*imgs_full])
        elif config.environment.part_screen_obs:
            observation_img = np.stack([*imgs_part])
    
    info = {'on_platform':on_platform, 'djump':djump}
    
    return observation_img, reward, terminal, info
    
# TODO измените загрузку изображений по образцу структуры выше 
class EdgesDataset(datasets.ImageFolder):
    '''Датасет возвращает пару двухканальных изображений следующей структуры:
    [img(timestep-3), img(timestep)], [img_part(timestep-3), img_part(timestep)], timestep, (player_x, player_y), (walk_distance, jump_height), reward
    где timestep - момент времени изображения self.samples[index],
    img - преобразованное изображение,
    img_part - преобразованный обрезанный вокруг кида кусок изображения
    reward - суммарная награда за время [t, t+timestep_stride]
    player_x, player_y - координаты кида
    walk_distance, jump_height - сумма действий, предпринимаемых на [t,t+timestep_stride] шагах
    '''
    
    
    def __init__(self, config):
        self.config = config.parameters.edges_dataset
        
        super(EdgesDataset, self).__init__(root=self.config.data_root, is_valid_file=None)
        
        self.config = config.parameters.edges_dataset
        
        self.config.image_size = config.environment.image_size
        self.config.timestep_stride = config.environment.timestep_stride
        self.config.timestep_max = config.environment.timestep_max
        
        self.zero_screenshot = zero_screenshot_load(self.config.zero_screenshot_path, self.config.image_size, self.config.pad, self.config.aperture_size)
        
        # Временно оставили None как заглушку
        self.zero_screenshot_part = None

    def __getitem__(self, index):

        image_path_1, _ = self.samples[index]
        image_path_2 = re.sub(r'/screenshots/', r'/screenshots_add/', image_path_1)
        
        metadata_path = image_path_2[:-3] + 'txt'
        
        timestep, reward, player_x, player_y, walk_distance, jump_height, on_platform, djump = extract_metadata(metadata_path)
        
        if timestep < config.environment.timestep_max:
            timestep_next = timestep + config.environment.timestep_stride
            terminal = False
        else:
            terminal = True
        
        # do your magic here
        # флаг 0 = читаем черно-белое изображение
        img1 = cv2.imread(image_path_1, 0)
        img2 = cv2.imread(image_path_2, 0)
        
        img_full1 = transform_img(img1, dim=[self.config.image_size, self.config.image_size], zero_screenshot=self.zero_screenshot, aperture_size=self.config.aperture_size)
        img_full2 = transform_img(img2, dim=[self.config.image_size, self.config.image_size], zero_screenshot=self.zero_screenshot, aperture_size=self.config.aperture_size)
        
        img_part1 = transform_img(img1, dim=[self.config.image_size, self.config.image_size], zero_screenshot=self.zero_screenshot_part, part_size=self.config.part_size, player_x=player_x, player_y=player_y, aperture_size=self.config.aperture_size)
        img_part2 = transform_img(img2, dim=[self.config.image_size, self.config.image_size], zero_screenshot=self.zero_screenshot_part, part_size=self.config.part_size, player_x=player_x, player_y=player_y, aperture_size=self.config.aperture_size)
        
        sample_full = cv2.merge([img_full1, img_full2])
        sample_part = cv2.merge([img_part1, img_part2])
        
        return (transforms.ToTensor()(sample_full), transforms.ToTensor()(sample_part), timestep, player_x, player_y), (walk_distance, jump_height), reward, terminal
    
    def load_timestep(self, seed, timestep):
        
        image_path = self.config.data_root + f'/screenshots0000_{seed}/{timestep}.png'
        
        return load_image_obs(image_path)
    
    def load_episodes(self, seeds):
        
        if isinstance(seeds, list):
            timestep_list = []
            for seed in seeds:
                timestep_list.extend([self.load_timestep(seed, timestep) for timestep in np.arange(self.config.timestep_stride, self.config.timestep_max, self.config.timestep_stride)])
        elif isinstance(seeds, int):
            timestep_list = [self.load_timestep(seeds, timestep) for timestep in np.arange(self.config.timestep_stride, self.config.timestep_max, self.config.timestep_stride)]
        else:
            raise Exception
            
        observation_imgs = np.stack(list(map(lambda x: x[0],timestep_list)))
        actions = np.stack(list(map(lambda x: x[1],timestep_list)))
        rewards = np.stack(list(map(lambda x: x[2],timestep_list)))
        terminals = np.stack(list(map(lambda x: x[3],timestep_list)))
        
        return observation_imgs, actions, rewards, terminals
    
    
def zero_screenshot_to_device(zero_screenshot_path, image_size, pad, aperture_size, device):
#     '''Загружаем нулевой скриншот с указанными параметрами на нужный нам девайс и детатчим градиенты'''
#     zero_screenshot_path = 'Sadistic_Music_Factory_screenshots/Sadistic_Music_Factory/Zero_screenshot.png'
    zero_screenshot = cv2.imread(zero_screenshot_path, 0)
    zero_screenshot = fullscreen_transform(zero_screenshot, dim=[image_size,image_size], pad=pad, aperture_size=aperture_size)
    zero_screenshot_tensor = torch.Tensor(zero_screenshot).to(device).detach()