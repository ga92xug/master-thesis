import random
from typing import Any, Dict, List, Tuple, Union

from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds import get_dataset

from sklearn.model_selection import train_test_split
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data.dataset import Dataset, random_split

from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    STL10,
    MNIST,
    ImageNet
)

import sys
import os
sys.path.append(f"{os.getcwd()}")

from src.data.datasets.utils import get_normalize_weights
from src.data._transforms.get_transforms import get_transforms


def create_datasets(
    # data
    data_dir: str,
    name: str,
    # transforms
    resolution: int,
    augment: Union[bool, Dict[str, Any]],
    channel_wise_mean_images: List[float],
    channel_wise_std_images: List[float],
    # dataset
    valid_size: float,
    should_normalize_weights: bool,
    reduction_factor: float,
    # download
    download: bool = False,
    **kwargs,
) -> Tuple[Dict[str, Dataset], torch.Tensor, Dict[str, Any]]:

    location = os.path.join(data_dir, name)
    dataset = get_dataset('camelyon17', root_dir=location, download=False)


    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images,
        verbose=1,
    )
        
    train_dataset = dataset.get_subset("train",transform=train_transform)
    valid_dataset = dataset.get_subset("val",transform=valid_transform)
    test_dataset = dataset.get_subset("test",transform=valid_transform)

    # Wrap the datasets
    train_dataset = Dataset_Wrapper(train_dataset)
    valid_dataset = Dataset_Wrapper(valid_dataset)
    test_dataset = Dataset_Wrapper(test_dataset)

    _datasets = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

    dataloader_kwargs = {}
    normalized_weights = None


    return _datasets, normalized_weights, dataloader_kwargs



class Dataset_Wrapper(Dataset):
    """
    This class is a wrapper for the WILDS datasets. Since we don't need the metadata, we can just return the x and y.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        x, y, meta = self.dataset[index]
        return x, y
        
    def __len__(self):
        return len(self.dataset)
