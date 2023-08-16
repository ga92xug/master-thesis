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
from experiment.datasets.utils import get_normalize_weights
#from experiment import dataloader


def build_loaders(
    images, 
    labels,
    transform,
    batch_size,
    eval_batch_size,
    workers,
):
    random_seed = 42

    # Split the data
    train_images, val_test_images, train_labels, val_test_labels = train_test_split(*[images, labels], test_size=0.20, random_state=random_seed, stratify=labels)
    test_images, val_images, test_labels, val_labels = train_test_split(*[val_test_images, val_test_labels] , test_size=0.5, random_state=random_seed, stratify=val_test_labels)

    # Create the DataLoaders
    train_loader = DataLoader(Custom_Dataset(train_images, train_labels, transform=transform), batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(Custom_Dataset(val_images, val_labels, transform=transform), batch_size=eval_batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(Custom_Dataset(test_images, test_labels, transform=transform), batch_size=eval_batch_size, shuffle=False, num_workers=workers)

    dataloaders = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }

    return dataloaders


def get_loaders(
    data_dir: str,
    name: str,
    resolution: int,
    should_normalize_weights: bool,
    channel_wise_mean_images: list,
    channel_wise_std_images: list,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    augment: bool,
    **kwargs,
):
    location = data_dir + name
    dataset = get_dataset('iwildcam', root_dir=location, download=False)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_wise_mean_images, std=channel_wise_std_images),
    ])

    # normalize weights
    normalized_weights = None

    train_data = dataset.get_subset("train",transform=transform)
    val_data = dataset.get_subset("val",transform=transform)
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size)
    val_loader = get_eval_loader("standard", val_data, batch_size=eval_batch_size)

    dataloaders = {
        "train": train_loader,
        "valid": val_loader,
        "test": val_loader,
    }

    return dataloaders, normalized_weights


@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    #print(cfg)
    dataloaders, normalize_weights = hydra.utils.call(cfg.training.dataset)

    train_dataloader = dataloaders["train"]
    valid_dataloader = dataloaders["valid"]
    test_dataloader = dataloaders["test"]

    max_val = 0
    for i, (images, labels) in enumerate(train_dataloader):
        print(images.shape)
        print(labels)

        break
    

if __name__ == "__main__":
    main()