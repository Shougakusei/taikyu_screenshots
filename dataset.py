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
from matplotlib import pyplot as plt

from torchsummary import summary

from abc import abstractmethod

def extract_metadata(fpath):
    
    with open(fpath, 'r') as file:
        info = file.read().rstrip()
        
    info_splited = info.split('_')
    
    timestep = int(info_splited[0])
    reward = int(info_splited[1])
    player_x = int(info_splited[2])
    player_y = int(info_splited[3])
    
    
    movements = info_splited[4].split(';')[:-1]
    
    walk_distance = jump_height = jump_true = 0
    
    for i in range(len(movements)):
        
        movement_list = [int(x) for x in list(movements[i])]
        
        if movement_list[0] == 1:
            walk_distance -= 1
            
        if movement_list[1] == 1:
            walk_distance += 1
        
        if movement_list[2] == 1:
            jump_true = 1
        
        # Если jumpt_true == 0, то не делаем ничего
        if (movement_list[3] == 0) and (jump_true == 1):
            jump_height += 1
        elif (movement_list[3] == 1) and (jump_true == 1):
            jump_height += 1
            jump_true = 0
        
    return timestep, reward, player_x, player_y, walk_distance, jump_height

def take_screen_part(img, player_x, player_y, width, height, pad):
    
    img = np.pad(img,pad)
    return img[pad+player_y-height:pad+player_y+height,pad+player_x-width:pad+player_x+width]

def fullscreen_transform(img, dim, pad = 32, zero_screenshot=None, aperture_size=7):
    
    img = img[pad:-pad, pad:-pad]
    img = cv2.resize(img, dim)
    img = cv2.Canny(img,150,250, apertureSize=aperture_size)
    if zero_screenshot is not None:
            img = (img != zero_screenshot) * img
    return img

def partscreen_transform(img, dim, zero_screenshot=None, aperture_size=7):
    
    img = cv2.resize(img, dim)
    img = cv2.Canny(img,150,250, apertureSize=aperture_size)
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

def zero_screenshot_load(zero_screenshot_path, image_size, pad, aperture_size):
    '''Загружаем и трансформируем нулевой скриншот с указанными параметрами'''
    zero_screenshot = cv2.imread(zero_screenshot_path, 0)
    zero_screenshot = fullscreen_transform(zero_screenshot, dim=[image_size,image_size], pad=pad, aperture_size=aperture_size)
    return zero_screenshot

class EdgesDataset(datasets.ImageFolder):
    '''Датасет возвращает пару двухканальных изображений следующей структуры:
    [img(timestep-3), img(timestep)], [img_part(timestep-3), img_part(timestep)], timestep, (player_x, player_y), (walk_distance, jump_height), reward
    где timestep - момент времени изображения self.samples[index],
    img - преобразованное изображение,
    img_part - преобразованный обрезанный вокруг кида кусок изображения
    reward - суммарная награда за время [t, t+5]
    player_x, player_y - координаты кида
    walk_distance, jump_height - сумма действий, предпринимаемых на [t,t+5] шагах
    '''
    
    
    def __init__(self, config):
        self.config = config.parameters.edges_dataset
        
        super(EdgesDataset, self).__init__(root=self.config.data_root, is_valid_file=None)
        
        self.config = config.parameters.edges_dataset
        
        self.config.image_size = config.environment.image_size
        
        self.zero_screenshot = zero_screenshot_load(self.config.zero_screenshot_path, self.config.image_size, self.config.pad, self.config.aperture_size)
        
        # Временно оставили None как заглушку
        self.zero_screenshot_part = None

    def __getitem__(self, index):

        image_path_1, _ = self.samples[index]
        image_path_2 = re.sub(r'/screenshots/', r'/screenshots_add/', image_path_1)
        
        metadata_path = image_path_2[:-3] + 'txt'
        
        timestep, reward, player_x, player_y, walk_distance, jump_height = extract_metadata(metadata_path)
        
        # do your magic here
        # флаг 0 = читаем черно-белое изображение
        img1 = cv2.imread(image_path_1, 0)
        img2 = cv2.imread(image_path_2, 0)
        
        img_full1 = transform_img(img1, dim=[self.config.image_size, self.config.image_size], zero_screenshot=self.zero_screenshot, aperture_size=self.config.aperture_size)
        img_full2 = transform_img(img2, dim=[self.config.image_size, self.config.image_size], zero_screenshot=self.zero_screenshot, aperture_size=self.config.aperture_size)
        
        img_part1 = transform_img(img1, dim=[self.config.image_size, self.config.image_size], zero_screenshot=self.zero_screenshot_part, part_size=self.config.part_size, player_x=player_x, player_y=player_y, aperture_size=self.config.aperture_size)
        img_part2 = transform_img(img2, dim=[self.config.image_size, self.config.image_size], zero_screenshot=self.zero_screenshot_part, part_size=self.config.part_size, player_x=player_x, player_y=player_y, aperture_size=self.config.aperture_size)
        
        sample = cv2.merge([img_full1, img_full2])
        sample_part = cv2.merge([img_part1, img_part2])
        
        
        
        return transforms.ToTensor()(sample), transforms.ToTensor()(sample_part), timestep, (player_x, player_y), (walk_distance, jump_height), reward 
    
def zero_screenshot_to_device(zero_screenshot_path, image_size, pad, aperture_size, device):
#     '''Загружаем нулевой скриншот с указанными параметрами на нужный нам девайс и детатчим градиенты'''
#     zero_screenshot_path = 'Sadistic_Music_Factory_screenshots/Sadistic_Music_Factory/Zero_screenshot.png'
    zero_screenshot = cv2.imread(zero_screenshot_path, 0)
    zero_screenshot = fullscreen_transform(zero_screenshot, dim=[image_size,image_size], pad=pad, aperture_size=aperture_size)
    zero_screenshot_tensor = torch.Tensor(zero_screenshot).to(device).detach()