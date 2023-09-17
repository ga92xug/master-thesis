import dis
from importlib import metadata
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image
import hydra
from omegaconf import DictConfig
#import cv2

from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds import get_dataset
import numpy as np

import sys
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
sys.path.append(f"{os.getcwd()}")
#print("current working directory: ", os.getcwd())
from training.datasets.utils import get_normalize_weights, get_transforms
#from experiment import dataloader

def get_loaders(
    data_dir: str,
    name: str,
    resolution: int,
    channel_wise_mean_images: list,
    channel_wise_std_images: list,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    augment: bool,
    distribution_shift: bool,
    **kwargs,
):
    location = data_dir + name
    dataset = get_dataset(name, root_dir=location, download=False)

    # Define the transformations
    train_transform = get_transforms(resolution, augment, channel_wise_mean_images, channel_wise_std_images)
    valid_transform = get_transforms(resolution, False, channel_wise_mean_images, channel_wise_std_images)

    # normalize weights
    normalized_weights = None

    train_data = dataset.get_subset("train",transform=train_transform)    
    # there is an indistribution validation set if we don't want to use the distribution shift
    val_data = dataset.get_subset("val" if distribution_shift else "id_val",transform=valid_transform)
    test_data = dataset.get_subset("test",transform=valid_transform)
    # shuffle the train data is automatically done in the wilds dataloader
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size, num_workers=workers)
    val_loader = get_eval_loader("standard", val_data, batch_size=eval_batch_size, num_workers=workers)
    test_loader = get_eval_loader("standard", test_data, batch_size=eval_batch_size, num_workers=workers)

    dataloaders = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }

    return dataloaders, normalized_weights

