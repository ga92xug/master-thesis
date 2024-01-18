from copy import deepcopy
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(f"{os.getcwd()}")
os.environ['HYDRA_FULL_ERROR'] = '1'
from . import own_transforms, autoaugment


def get_transforms(
        resolution: int,
        augment: Union[bool, Dict],
        channel_wise_mean_images: List[float],
        channel_wise_std_images: List[float],
        verbose: int,
    ) -> Tuple[transforms.Compose, transforms.Compose]:

    original_augment = deepcopy(augment)
    train_transform = get_one_transform(resolution, augment, channel_wise_mean_images, channel_wise_std_images, validation=False)
    augment = deepcopy(original_augment)
    valid_transform = get_one_transform(resolution, augment, channel_wise_mean_images, channel_wise_std_images, validation=True)
    
    if verbose > 1:
        print("train_transform: ", train_transform)
        print("valid_transform: ", valid_transform)
    return train_transform, valid_transform


def get_one_transform(
    resolution: int,
    augment: Union[bool, Dict],
    channel_wise_mean_images: List[float],
    channel_wise_std_images: List[float],
    validation: bool = False,
    ) -> transforms.Compose:
    """
    Standard transforms for images. Augmentations can be passed as a dictionary.
    """
    if isinstance(augment, bool):
        transform_list = [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ]
        if channel_wise_mean_images is not None and channel_wise_std_images is not None:
            transform_list.extend([
                transforms.Normalize(
                    mean=channel_wise_mean_images,
                    std=channel_wise_std_images,
                )
            ])
        return transforms.Compose(transform_list)

    ###############################################
    # from here on, augment is a dictionary

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
    if "RandomResizedCrop" in augment.keys() and not validation:
        kwargs = augment.pop("RandomResizedCrop")
        transform_list.append(
            transforms.RandomResizedCrop(
                size=resolution,
                **kwargs,
            )
        )
    elif "short_side_center_crop" in augment.keys():
        transform_list.extend(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
            ]
        )
    elif "NoResize" in augment.keys():
        pass
    else:
        transform_list.append(transforms.Resize(resolution))

    # pop all size augmentations
    augment.pop("RandomResizedCrop", None)
    augment.pop("short_side_center_crop", None)
    augment.pop("NoResize", None)

    # add augmentations
    if isinstance(augment, dict):
        for key, value in augment.items():
            if "Own" in key:
                # random rotation does not allow for discrete choices
                transform_list.append(getattr(own_transforms, key)(**value))
            elif "CIFAR10Policy" in key:
                transform_list.append(getattr(autoaugment, key)(**value))
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