import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image
from omegaconf import DictConfig, OmegaConf
#import cv2

import h5py
import numpy as np

import sys
import os
sys.path.append(f"{os.getcwd()}")
os.environ['HYDRA_FULL_ERROR'] = '1'
from training.datasets.utils import get_normalize_weights, get_transforms


def split_without_stratify(images, labels, random_seed):
    """
    Not used anymore. Only to reproduce old results.
    """
    # Split the data into train, val, and test arrays.
    np.random.seed(random_seed)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    split = int(len(images) * 0.8)
    train_indices = indices[:split]
    val_indices = indices[split:split + int(len(images) * 0.1)]
    test_indices = indices[split + int(len(images) * 0.1):]

    train_images = images[train_indices]
    train_labels = labels[train_indices]
    val_images = images[val_indices]
    val_labels = labels[val_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def split_with_stratify(images, labels, random_seed, reduction_factor=None):
    if reduction_factor is None or reduction_factor == 1.0:
        train_images, val_test_images, train_labels, val_test_labels = train_test_split(*[images, labels], test_size=0.20, random_state=random_seed, stratify=labels)
        test_images, val_images, test_labels, val_labels = train_test_split(*[val_test_images, val_test_labels] , test_size=0.5, random_state=random_seed, stratify=val_test_labels)
    else:
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(*[images, labels], train_size=reduction_factor, random_state=random_seed, stratify=labels)
        train_images, val_images, train_labels, val_labels = train_test_split(*[train_val_images, train_val_labels], train_size=0.6, test_size=0.4, random_state=random_seed, stratify=train_val_labels)
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

class Custom_Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        assert len(images) == len(labels), "images and labels should have the same length"
        assert isinstance(images[0], np.ndarray) or isinstance(images[0], str), "images should be a list of numpy arrays or a list of strings"

        self.images = images
        self.labels = labels
        self.transform = transform

        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            raise RuntimeError("Unknown image type")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.int64)
        return image, label

def build_loaders(
    images, 
    labels,
    train_transform,
    valid_transform,
    batch_size,
    eval_batch_size,
    workers,
    split_function,
    reduction_factor=1.0,
):
    random_seed = 42
    # Split the data into train, val, and test arrays.
    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_function(images, labels, random_seed=random_seed, reduction_factor=reduction_factor)

    # Create the DataLoaders
    train_loader = DataLoader(Custom_Dataset(train_images, train_labels, transform=train_transform), batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(Custom_Dataset(val_images, val_labels, transform=valid_transform), batch_size=eval_batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(Custom_Dataset(test_images, test_labels, transform=valid_transform), batch_size=eval_batch_size, shuffle=False, num_workers=workers)

    dataloaders = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }

    return dataloaders


def get_Galaxy10_DECals(
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
    assert resolution <= 256, "The maximum resolution for Galaxy10_DECals is 256"

    location = data_dir + name + "/Galaxy10_DECals.h5"
    #print("location: ", location)
    with h5py.File(location, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    images = images.astype(np.uint8)

    # Define the transformations
    train_transform = get_transforms(resolution, augment, channel_wise_mean_images, channel_wise_std_images)
    valid_transform = get_transforms(resolution, False, channel_wise_mean_images, channel_wise_std_images)

    # normalize weights
    normalized_weights = get_normalize_weights(labels) if should_normalize_weights else 1

    dataloaders = build_loaders(images, labels, train_transform, valid_transform, batch_size, eval_batch_size, workers, split_with_stratify)
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
    reduction_factor=None,
    **kwargs,
):
    assert resolution <= 450, "The maximum resolution for ISIC_2019 is 450x450 since the minimum height is 450"

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
    
    # Define the transformations
    train_transform = get_transforms(resolution, augment, channel_wise_mean_images, channel_wise_std_images)
    valid_transform = get_transforms(resolution, False, channel_wise_mean_images, channel_wise_std_images)

    # normalize weights
    normalized_weights = get_normalize_weights(labels) if should_normalize_weights else 1

    dataloaders = build_loaders(images, labels, train_transform, valid_transform, batch_size, eval_batch_size, workers, split_with_stratify, reduction_factor=reduction_factor)
    return dataloaders, normalized_weights

