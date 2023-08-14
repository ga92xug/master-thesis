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

import h5py
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


def get_wilds(
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
    location = data_dir + name + "/Galaxy10_DECals.h5"
    #print("location: ", location)
    with h5py.File(location, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    images = images.astype(np.uint8)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_wise_mean_images, std=channel_wise_std_images),
    ])

    # normalize weights
    if should_normalize_weights:
        normalized_weights = get_normalize_weights(labels)
    else:
        # to gather the weighted accuracy
        normalized_weights = 1

    dataloaders = build_loaders(images, labels, transform, batch_size, eval_batch_size, workers)
    return dataloaders, normalized_weights


def get_ISIC_2019(
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
    # these files you download
    ground_truth = location + '/ISIC_2019_Training_GroundTruth.csv'
    images = location + '/ISIC_2019_Training_Input'

    df = pd.read_csv(ground_truth)
    
    for label in df.columns[1:]:
        df.loc[df[label] == 1.0, 'label'] = label


    #create instance of label encoder
    lab = LabelEncoder()
    df['label'] = lab.fit_transform(df['label'])
        
    df.rename(columns={'image': 'name'}, inplace=True)
    df['name'] = df['name'].apply(lambda x: "{}/{}.jpg".format(images,x))
    df = df[['name', 'label']]
    labels = df['label'].values
    images = df['name'].values
    #images = df['name'].apply(lambda file_location: np.array(Image.open(file_location)).astype(np.uint8)).values
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_wise_mean_images, std=channel_wise_std_images),
    ])

    # normalize weights
    if should_normalize_weights:
        normalized_weights = get_normalize_weights(labels)
    else:
        # to gather the weighted accuracy
        normalized_weights = 1

    dataloaders = build_loaders(images, labels, transform, batch_size, eval_batch_size, workers)
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