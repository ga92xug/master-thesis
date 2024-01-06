
import re
from typing import Dict, List, Tuple, Union
from sklearn.model_selection import train_test_split
from sympy import root
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
from omegaconf import DictConfig

import medmnist
from medmnist import INFO, Evaluator
#from medmnist import INFO, Evaluator

import sys
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
sys.path.append(f"{os.getcwd()}")
from src.data.datasets.utils import get_normalize_weights, get_transforms


def create_datasets(
    # data
    data_dir: str,
    name: str,
    # transforms
    resolution: int,
    augment: Union[bool, Dict],
    channel_wise_mean_images: List,
    channel_wise_std_images: List,
    # dataset
    should_normalize_weights: bool,
    **kwargs,
) -> Tuple[Tuple[Dataset, Dataset, Dataset], torch.Tensor, callable]:
    
    location = data_dir + "medmnist/"

    DataClass = getattr(medmnist, name)

    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images,
        verbose=1,
    )
    
    train_set = DataClass(split='train', transform=train_transform, download=True, root=location)
    val_set = DataClass(split='val', transform=valid_transform, download=True, root=location)
    test_set = DataClass(split='test', transform=valid_transform, download=True, root=location)

    # normalize weights
    train_labels = train_set.labels.squeeze()
    normalized_weights = get_normalize_weights(train_labels) if should_normalize_weights else 1

    datasets = {
        "train": train_set,
        "valid": val_set,
        "test": test_set,
    }

    dataloader_kwargs = {
        "train": {
            "collate_fn": custom_collate,
        },
        "valid": {
            "collate_fn": custom_collate,
        },
        "test": {
            "collate_fn": custom_collate,
        },
    }

    return datasets, normalized_weights, dataloader_kwargs


def custom_collate(batch):
    data, labels = zip(*batch)
    data = torch.stack(data)
    assert len(labels[0]) == 1, "Assuming label is wrapped as a single-element list"
    
    labels = torch.tensor(np.array(labels), dtype=torch.int64)
    labels = labels.squeeze()
    return data, labels