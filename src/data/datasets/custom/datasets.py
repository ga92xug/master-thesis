from typing import Dict, List, Union

import hydra
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from src.data.datasets.utils import (
    get_normalize_weights, 
    get_transforms, 
    split_with_stratify, 
)

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
        #print("image: ", image)
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


def create_datasets(
    # images and labels
    data: tuple,    
    # transforms
    resolution: int,
    augment: Union[bool, Dict],
    channel_wise_mean_images: List,
    channel_wise_std_images: List,
    # dataset
    val_size: float,
    test_size: float,
    reduction_factor: float,
    should_normalize_weights: bool,
    # just test
    test_as_valid: bool = False,
    **kwargs,
    ):

    # we use recursive instantiation from hydra to get the data
    images, labels = data

    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        original_augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images,
        verbose=1,
    )

    if isinstance(images, dict):
        # split the data; assume there are train and test sets; create val set from train set if not exists
        if "test" not in images:
            raise RuntimeError("images should have a key named 'test' when images is a dictionary.")
        train_images = images["train"]
        train_labels = labels["train"]
        test_images = images["test"]
        test_labels = labels["test"]

        val_images = images.get("val", None)
        val_labels = labels.get("val", None)

        if val_images is None:
            assert val_size > 0.0, "val_size should be greater than 0.0 when images is a dictionary and val_images is None."
            # Split the data into train, val
            train_images, train_labels, val_images, val_labels, _, _ = split_with_stratify(
                images=train_images, 
                labels=train_labels, 
                reduction_factor=reduction_factor,
                val_size=val_size,
                test_size=test_size,
            )
        else:
            # here we only do reduction_factor on train set
            train_images, train_labels, _, _, _, _ = split_with_stratify(
                images=train_images, 
                labels=train_labels, 
                reduction_factor=reduction_factor,
                val_size=0.0,
                test_size=0.0,
            )
    else:
        # Split the data into train, val, and test arrays.
        train_images, train_labels, val_images, val_labels, test_images, test_labels = \
            split_with_stratify(
                images=images, 
                labels=labels, 
                reduction_factor=reduction_factor,
                val_size=val_size,
                test_size=test_size,
            )

    if test_as_valid:
        # swap val_loader and test_loader to test generalization early
        val_images, test_images = test_images, val_images

    # Create the DataLoaders
    train_set = Custom_Dataset(train_images, train_labels, transform=train_transform)
    val_set = Custom_Dataset(val_images, val_labels, transform=valid_transform)
    test_set = Custom_Dataset(test_images, test_labels, transform=valid_transform)

    # normalize weights
    normalized_weights = get_normalize_weights(train_labels, 1) if should_normalize_weights else 1

    return train_set, val_set, test_set, normalized_weights