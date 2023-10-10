from ast import Tuple
from calendar import c
from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(f"{os.getcwd()}")
os.environ['HYDRA_FULL_ERROR'] = '1'
from training.datasets import own_transforms

def get_normalize_weights(
        labels: List or np.ndarray,
    ):
    """
    Normalize the weights of the dataset based on the labels.
    """
    df = pd.DataFrame({"label": labels})
    weights = df['label'].value_counts() / df['label'].value_counts().sum()
    # sort based on index
    weights = weights.sort_index()
    assert np.isclose(weights.sum(), 1.0), "weights should sum to 1.0"
    weights = weights.values
    print("weights: ", weights.tolist())
    return weights


def get_transforms(
        resolution: int,
        original_augment: bool or dict,
        channel_wise_mean_images: list,
        channel_wise_std_images: list,
    ) -> Tuple(transforms.Compose, transforms.Compose):

    augment = deepcopy(original_augment)
    train_transform = get_one_transform(resolution, augment, channel_wise_mean_images, channel_wise_std_images, validation=False)
    augment = deepcopy(original_augment)
    valid_transform = get_one_transform(resolution, augment, channel_wise_mean_images, channel_wise_std_images, validation=True)
    
    print("train_transform: ", train_transform)
    print("valid_transform: ", valid_transform)
    return train_transform, valid_transform


def get_one_transform(
    resolution: int,
    augment: bool or dict,
    channel_wise_mean_images: list,
    channel_wise_std_images: list,
    validation: bool = False,
    ) -> transforms.Compose:
    """
    Standard transforms for images. Augmentations can be passed as a dictionary.
    """
    if isinstance(augment, DictConfig):
        augment = OmegaConf.to_container(
            augment, resolve=True, throw_on_missing=True
        )

    # which augmentations to use
    if validation:
        augment = augment.get("all", {})
    else:
        augment = {**augment.get("all", {}), **augment.get("train", {})} 

    transform_list = []

    # resize
    if isinstance(augment, dict) and "short_side_center_crop" in augment.keys():
        transform_list.extend(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
            ]
        )
        augment.pop("short_side_center_crop")
    else:
        transform_list.append(transforms.Resize(resolution))

    # add augmentations
    if isinstance(augment, dict):
        for key, value in augment.items():
            if "Own" in key:
                # random rotation does not allow for discrete choices
                transform_list.append(getattr(own_transforms, key)(**value))
            else:
                transform_list.append(getattr(transforms, key)(**value))
    elif isinstance(augment, bool):
        NotImplementedError("Bool Augmentation is not implemented yet")
    else:
        raise RuntimeError("Unknown Augmentation Type")

    transform_list.extend([transforms.ToTensor()])

    # normalize
    if channel_wise_mean_images is not None and channel_wise_std_images is not None:
        transform_list.extend([
            transforms.Normalize(
                mean=channel_wise_mean_images,
                std=channel_wise_std_images,
            )
        ])

    return transforms.Compose(transform_list)