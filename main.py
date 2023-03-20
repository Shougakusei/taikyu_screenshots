import os
import re

import torch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
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


import argparse
from dataset import EdgesDataset
from torch.utils.data import DataLoader
from utils.utils import load_config

def main(config_file):
    
    config = load_config(config_file)
    
    device = torch.device(config.operation.device if torch.cuda.is_available() else "cpu")
    
    train_ds = EdgesDataset(
    root=dataroot,
    dim=[image_size,image_size],
    zero_screenshot=zero_screenshot,
    aperture_size=aperture_size,
    part_size=part_size
    )

    dataloader = DataLoader(train_ds, batch_size, shuffle=True)

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="palladium-smf.yml",
        help="config file to run(default: palladium-smf.yml)",
    )