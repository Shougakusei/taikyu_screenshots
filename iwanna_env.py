import gym
from gym import spaces
import numpy as np
import torch
from torch import nn
from win32com.client import Dispatch
import subprocess
import os
import signal
import time
import shutil
import yaml
from attrdict import AttrDict
from utils import find_file, load_config
from dataset import load_image_obs, zero_screenshot_load, zero_screenshot_part_load

def wait_path_exists(path, sleep_time=0.25, peridos_to_wait=40):
    time_counter = 0
    while (not os.path.exists(path)) or os.path.exists(f'C:/Users/user/Taikyu_project/stuff/Super Fish/Realtime/gamemaker_flg.txt'):
        time.sleep(sleep_time)
        time_counter += 1
        if time_counter > peridos_to_wait:
            break
            
def start_iwanna(folder, exe):
    os.chdir(folder)
    return subprocess.Popen([exe])

def close_iwanna(process):
    try:
        os.kill(process.pid, signal.SIGTERM)
    except PermissionError:
        print('Отказано в доступе при попытке остановки процесса')
    except OSError:
        print('Процесс не существует')
    
def actions_to_controls(walk_length, jump_height, on_platform, djump):
    
    controls_array = np.zeros([6,4], dtype=np.int8)
    
    walk_length = round(walk_length)
    jump_height = round(jump_height)
    
    if on_platform or djump:
        jump_true = 0
    else:
        jump_true = 1
    
    for i in range(6):
        
        if walk_length > 0:
            
            controls_left = 0
            controls_right = 1
                
            walk_length -= 1
        
        elif walk_length < 0:
            
            controls_left = 1
            controls_right = 0
            
            walk_length += 1
        
        else:
            
            controls_left = 0
            controls_right = 0
            
        if jump_height > 0 and jump_true == 0:
            
            controls_jump = 1
            
            jump_true = 1
            jump_height -= 1
        
        else:
            
            controls_jump = 0
            
            jump_height -= 1
            
        if jump_height <= 0 and jump_true == 1:
            
            controls_jump_release = 1
            
            jump_true = 0
        
        else:
            
            controls_jump_release = 0
            
        controls_array[i] = [controls_left,controls_right,controls_jump,controls_jump_release]
    
    return controls_array

def write_controls_txt(controls, file_path, rewrite=True):
    
    if rewrite:
        open(file_path, 'w').close()
    
    for control in controls:
        with open(file_path, 'a') as f:
            f.write(''.join(str(x) for x in control) + '\n')

def read_start_conditions(file_path):
    with open(file_path) as file:
        lines = [line.rstrip() for line in file]
    seed = int(lines[0])
    timeline_start = int(lines[1])
    player_x_start = float(lines[2])
    player_y_start = float(lines[3])
    return seed, timeline_start, player_x_start, player_y_start

def write_start_conditions(file_path, seed, timeline_start, player_x_start, player_y_start, rewrite=True):
    
    if rewrite:
        open(file_path, 'w').close()
        
    with open(file_path, 'w') as file:
        file.write(f'{seed}\n')
        file.write(f'{timeline_start}\n')
        file.write(f'{player_x_start}\n')
        file.write(f'{player_y_start}\n')  
        
class PythonFlg():
    
    def __init__(self, flg_path):
        
        self.raised = 0
        self.flg_path = flg_path
        if os.path.exists(self.flg_path):
            os.remove(self.flg_path)

        
    def raise_flg(self):
        
        if self.raised == 0:
            self.raised = 1
            self.flg_file = open(self.flg_path, "w")
            
    def lower_flg(self):
        
        if self.raised == 1:
            
            self.flg_file.close()
            os.remove(self.flg_path)
            self.raised = 0
            
class IwannaEnv(gym.Env):
    
    def __init__(self, config, seed=None, timeline_start=None, player_x_start=None, player_y_start=None, round_reward = False, action_rescale = True):
        
        
        self.config = config
        self.flg = PythonFlg(self.config.directories.root_folder + 'Realtime/python_flg.txt')
        
        super(IwannaEnv, self).__init__()
        
        self.round_reward = round_reward
        self.action_rescale = action_rescale
        
        # каналы изображения - [доп скрин N full, ..., доп скрин 0 full, основной скрин full,
        #                       доп скрин N part, ..., доп скрин 0 part, основной скрин part]
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=((self.config.environment.add_screen_count + 1) * 
                                                   (self.config.environment.full_screen_obs + self.config.environment.part_screen_obs), 
                                                   self.config.environment.image_size, 
                                                   self.config.environment.image_size), 
                                            dtype=np.uint8)
        
        # Действия формата - (ходьба, где положительное значение - вправо, отрицательное - влево;
        #                     прыжок, где 0 - не прыгаем, 1 - прыгаем и отпускаем в первый же кадр,
        #                     ... 6 - прыгаем, и отпускаем на шестой кадр,
        #                         7 - не отпускаем кнопку прыжка,
        #                         8 - не отпускаем кнопку прыжка)
        #
        # При записи действий в txt идет округление до ближайшего десятичного знака
        self.action_low = np.array([-self.config.environment.timestep_stride, 0])
        self.action_high = np.array([self.config.environment.timestep_stride, self.config.environment.timestep_stride + 2])
        
        if self.action_rescale:
            self.action_space = spaces.Box(low=np.array([-1.0,-1.0]), high=np.array([1.0,1.0]), dtype=np.float32)
        else:                        
            self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        
        self.exe_folder = self.config.directories.exe_folder
        self.exe = self.config.directories.exe
        self.root_folder = self.config.directories.root_folder
        self.screenshot_folder = self.root_folder + 'Realtime/screenshots/'
        self.screenshot_add_folder = self.root_folder + 'Realtime/screenshots_add/'
        self.controls_folder = self.root_folder + 'Realtime/controls/'
        
        self.zero_screenshot = zero_screenshot_load(
            self.root_folder + 'Realtime/Zero_screenshot.png',
            image_size=self.config.environment.image_size,
            pad=self.config.parameters.edges_dataset.pad,
            aperture_size=self.config.parameters.edges_dataset.aperture_size)
        
        self.zero_screenshot_part = zero_screenshot_part_load(
            self.root_folder + 'Realtime/Zero_screenshot_part.png', 
            image_size=self.config.environment.image_size, 
            part_size=self.config.parameters.edges_dataset.part_size, 
            player_x=self.config.parameters.edges_dataset.zero_screenshot_player_x, 
            player_y=self.config.parameters.edges_dataset.zero_screenshot_player_y, 
            pad=self.config.parameters.edges_dataset.pad, 
            aperture_size=self.config.parameters.edges_dataset.aperture_size)
        
        if seed is None or timeline_start is None or player_x_start is None or player_y_start is None:
            self.seed, self.timeline_start, self.player_x_start, self.player_y_start = read_start_conditions(self.root_folder + r'Realtime\start_conditions.txt')
        else:
            self.set_start_conditions(seed, timeline_start, player_x_start, player_y_start)
        
        # Текстовый вид стартовой позиции таймлайна с лидирующими нулями для записи в названиях файлов
        self.timeline_start_str = (4 - len(str(self.timeline_start))) * '0' + str(self.timeline_start)

        self.timestep = self.config.environment.timestep_stride
        
        self.process = None
        
        self.on_platform = True
        self.djump = True
        
    
    def reset(self):
        '''Для корректного старта энвайромента после создания нужно сделать .reset()'''
        self.flg.raise_flg()
        
        if self.process is not None:
            close_iwanna(self.process)
        
        # Обнуляем текущее время энвайронмента
        self.timestep = self.config.environment.timestep_stride
        
        self.episode_screenshots_folder = self.screenshot_folder + f'screenshots{self.timeline_start_str}_{self.seed}/'
        
        self.episode_screenshots_add_folder = self.screenshot_add_folder + f'screenshots{self.timeline_start_str}_{self.seed}/'
        
        print(self.episode_screenshots_folder)
        
        self.episode_controls_folder = self.controls_folder + f'controls{self.timeline_start_str}_{self.seed}/'
        
        # Создаем новую папку для скриншотов
        if os.path.exists(self.episode_screenshots_folder):
            shutil.rmtree(self.episode_screenshots_folder)
        os.mkdir(self.episode_screenshots_folder)
        
        # Создаем новую папку для доп скриншотов
        if os.path.exists(self.episode_screenshots_add_folder):
            shutil.rmtree(self.episode_screenshots_add_folder)
        os.mkdir(self.episode_screenshots_add_folder)
        
        # Создаем новую папку для записи инпутов
        if os.path.exists(self.episode_controls_folder):
            shutil.rmtree(self.episode_controls_folder)
        os.mkdir(self.episode_controls_folder)
        
        write_controls_txt(actions_to_controls(0,0, True, True), self.episode_controls_folder + f'controls{0}.txt')
        
        img_path = self.episode_screenshots_folder + f'{self.timestep}.png'
        add_img_path = self.episode_screenshots_add_folder + f'{self.timestep}.png'
        
        self.flg.lower_flg()
        
        self.process = start_iwanna(self.exe_folder, self.exe)
    
        wait_path_exists(img_path)
        wait_path_exists(add_img_path)
        
        observation_img, _, _, _ = load_image_obs(img_path, self.zero_screenshot, self.zero_screenshot_part, self.config)
        
        return observation_img
    
    def step(self, action):
        
        self.flg.raise_flg()
        
        walk_length, jump_height = action
                                    
        if self.action_rescale:
            walk_length *= self.action_high[0]
            jump_height *= self.action_high[1]
            if jump_height < 0:
                jump_height = 0

        write_controls_txt(actions_to_controls(walk_length, jump_height, self.on_platform, self.djump), self.episode_controls_folder + f'controls{self.timestep}.txt')
        
        self.flg.lower_flg()
        
        img_path = self.episode_screenshots_folder + f'{self.timestep}.png'
        add_info_path = self.episode_screenshots_add_folder + f'{self.timestep}.txt'
        
        wait_path_exists(img_path)
        wait_path_exists(add_info_path)
        
        observation_img, reward, terminal, info = load_image_obs(img_path, self.zero_screenshot, self.zero_screenshot_part, self.config)
        
        if self.round_reward:
            reward = round(reward)
        
        self.on_platform = info['on_platform']
        self.djump = info['djump']
        
        self.timestep += self.config.environment.timestep_stride
        
        print(f'{round(walk_length,2), round(jump_height,2)} - {reward}')
        
        return observation_img, reward, terminal, info
    
    def render(self, mode='human'):
        
        pass
    
    def close(self):
        
        self.flg.lower_flg()
        close_iwanna(self.process)
    
    def __del__(self):
        
        self.flg.lower_flg()
    
    def set_start_conditions(self, seed, timeline_start, player_x_start, player_y_start):
        
        write_start_conditions(self.root_folder + 'Realtime/start_conditions.txt', seed, timeline_start, player_x_start, player_y_start)
        self.seed = seed
        self.timeline_start = timeline_start
        self.player_x_start = player_x_start
        self.player_y_start = player_y_start
 
