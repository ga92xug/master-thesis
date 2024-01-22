import random
from typing import Any, Dict, List, Tuple, Union

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
)

import sys
import os
sys.path.append(f"{os.getcwd()}")

from src.data.datasets.utils import get_normalize_weights
from src.data._transforms.get_transforms import get_transforms

def create_datasets(
    # data
    #data_dir: str,
    name: str,
    # transforms
    resolution: int,
    augment: Union[bool, Dict],
    channel_wise_mean_images: List,
    channel_wise_std_images: List,
    # dataset
    valid_size: float,
    should_normalize_weights: bool,
    reduction_factor: float,
    # download
    download: bool = False,
    **kwargs,
) -> Tuple[Dict[str, Dataset], torch.Tensor, Dict[str, Any]]:
    """
    Creates [train, valid, test] datasets from the predefined datasets.

    Returns: 
    Tuple[Dict[str, Dataset], torch.Tensor, Dict[Any]]: A tuple containing the datasets, the normalization weights and the dataloader kwargs.
    """

    data_dir = kwargs["data"]["data_dir"]
    
    assert name in ["cifar10", "cifar100", "stl10", "mnist"], "Unknown dataset name."

    #location = data_dir + name + "/"
    location = os.path.join(data_dir, name)

    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images,
        verbose=1,
    )

    dataset_class = getattr(datasets, name.upper())
    
    # load the dataset
    if "cifar" in name:        
        train_dataset = dataset_class(root=location, train=True, download=download, transform=None)
        valid_dataset = None
        test_dataset = dataset_class(root=location, train=False, download=download, transform=valid_transform)

    elif name == "stl10":
        train_dataset = dataset_class(root=location, split="train", download=download, transform=None)
        valid_dataset = None
        test_dataset = dataset_class(root=location, split="test", download=download, transform=valid_transform)

    elif name == "mnist":
        train_dataset = dataset_class(root=location, train=True, download=download, transform=None)
        valid_dataset = None
        test_dataset = dataset_class(root=location, train=False, download=download, transform=valid_transform)

    else:
        raise RuntimeError(f"Unknown dataset name: {name}.")

    if valid_dataset is None or reduction_factor < 1.0:
        valid_size = int(len(train_dataset) * valid_size)
        lengths = [len(train_dataset) - valid_size, valid_size]
        train_subset, val_subset = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))

        if reduction_factor < 1.0:
            # without a random seed -> random seed from global splits should be different 
            # randomly select a subset of the data
            reduction_size = int(len(train_subset) * reduction_factor)
            lengths = [reduction_size, len(train_subset) - reduction_size]
            train_subset, _ = random_split(train_subset, lengths)

        train_dataset = Subset_Transform_Dataset(train_subset, train_transform)
        valid_dataset = Subset_Transform_Dataset(val_subset, valid_transform)

    _datasets = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

    dataloader_kwargs = {}

    # Normalized weights
    # since we are using the stratified_subset_indices function, we can just use the train_val_dataset
    # Extract labels from the dataset
    labels = [label for _, label in train_dataset]
    normalized_weights = get_normalize_weights(labels) if should_normalize_weights else None

    return _datasets, normalized_weights, dataloader_kwargs


class Subset_Transform_Dataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def stratified_subset_indices(
    dataset: Dataset, 
    reduction_factor: float, 
    validation_split: float = 0.1, 
    random_seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Generates stratified train and validation indices for a dataset with an uneven distribution of classes,
    randomly excluding images based on the reduction factor.

    Args:
    dataset (Dataset): The dataset to be subset.
    reduction_factor (float): The fraction of the dataset to be used. Must be between 0 and 1.
    validation_split (float): The fraction of the dataset to be used as validation set. Must be between 0 and 1.
    random_seed (int): Seed for random number generator for reproducibility.

    Returns:
    Tuple[List[int], List[int]]: Lists of indices for training and validation subsets.

    # we can not just split the dataset as the transformations are different for train and valid
    # so we need to split the indices and then use the SubsetRandomSampler
    # train_idx, valid_idx = stratified_subset_indices(
    #     dataset=train_dataset, 
    #     reduction_factor=reduction_factor, 
    #     validation_split=valid_size,
    #     random_seed=42
    # )

    # might pose problem in DDP since DistributedSampler is used
    # if use maybe define a new dataset or split dataset
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    """

    assert 0 < reduction_factor <= 1, "reduction_factor must be between 0 and 1."
    assert 0 < validation_split < 1, "validation_split must be between 0 and 1."

    # Gathering indices for each class
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Reducing and splitting indices
    train_indices_all, val_indices_all = [], []
    for label, indices in class_indices.items():
        # Randomly selecting a subset of indices based on reduction factor
        reduced_indices = random.sample(indices, int(np.round(len(indices) * reduction_factor)))

        # Splitting into train and validation sets
        train_indices, val_indices = train_test_split(reduced_indices, test_size=validation_split, random_state=random_seed)
        train_indices_all.extend(train_indices)
        val_indices_all.extend(val_indices)

    # Checking for any overlap or duplicate indices
    assert len(set(train_indices_all)) == len(train_indices_all), "Duplicate indices in train set."
    assert len(set(val_indices_all)) == len(val_indices_all), "Duplicate indices in validation set."
    assert len(set(train_indices_all).intersection(val_indices_all)) == 0, "Overlap between train and validation indices."

    print("train_size: ", len(train_indices_all))
    print("val_size: ", len(val_indices_all))
    print("total_size: ", len(train_indices_all) + len(val_indices_all))

    return train_indices_all, val_indices_all