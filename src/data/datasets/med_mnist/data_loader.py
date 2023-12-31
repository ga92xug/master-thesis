
import re
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
from training.datasets.utils import get_normalize_weights, get_transforms

def custom_collate(batch):
    data, labels = zip(*batch)
    data = torch.stack(data)
    assert len(labels[0]) == 1, "Assuming label is wrapped as a single-element list"
    
    labels = torch.tensor(np.array(labels), dtype=torch.int64)
    labels = labels.squeeze()
    return data, labels

def build_loaders(
        batch_size,
        eval_batch_size,
        data_dir,
        name,
        channel_wise_mean_images,
        channel_wise_std_images,
        resolution,
        n_out_classes,
        reduction_factor=1.0,
        should_normalize_weights=True,
        workers=8,
        augment=False,
        test_as_valid: bool = False,
        **kwargs,
    ):
    
    location = data_dir + "medmnist/"

    DataClass = getattr(medmnist, name)

    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        original_augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images
    )
    
    train_dataset = DataClass(split='train', transform=train_transform, download=True, root=location)
    valid_dataset = DataClass(split='val', transform=valid_transform, download=True, root=location)
    test_dataset = DataClass(split='test', transform=valid_transform, download=True, root=location)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=custom_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=workers, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=workers, collate_fn=custom_collate)

    # normalize weights
    labels = train_dataset.labels.squeeze()
    normalized_weights = get_normalize_weights(labels) if should_normalize_weights else 1

    if test_as_valid:
        valid_loader = test_loader

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    
    return loaders, normalized_weights

