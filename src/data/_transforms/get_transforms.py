from copy import deepcopy
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import InterpolationMode

import sys
import os
sys.path.append(f"{os.getcwd()}")
os.environ['HYDRA_FULL_ERROR'] = '1'
from . import own_transforms, autoaugment, staincolorjitter


def get_transforms(
        resolution: int,
        augment: Union[bool, Dict],
        channel_wise_mean_images: List[float],
        channel_wise_std_images: List[float],
        verbose: int = 1,
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
    transform_list = []

    if isinstance(augment, bool):
        if augment:
            raise RuntimeError("Augmentations are not defined. Define them through a dictionary.")

        transform_list.extend([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

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
        augment = {**augment.get("all", {}), **augment.get("eval", {})}
    else:
        augment = {**augment.get("all", {}), **augment.get("train", {})} 

    interpolation = getattr(InterpolationMode, augment.pop("interpolation", "BILINEAR"))


    # resize
    if "RandomResizedCrop" in augment.keys():
        kwargs = augment.pop("RandomResizedCrop")
        if "size" not in kwargs.keys():
            # if size is not given, use resolution
            kwargs["size"] = resolution
        transform_list.append(transforms.RandomResizedCrop(**kwargs, interpolation=interpolation))
    elif "short_side_center_crop" in augment.keys():
        kwargs = augment.pop("short_side_center_crop", {})

        # if size is not given, use resolution
        # like this we can resize and crop to different sizes like in the torchvision classification script
        resize_size = kwargs.get("resize_size", resolution)
        crop_size = kwargs.get("crop_size", resolution)
        transform_list.extend([
                transforms.Resize(size=resize_size, interpolation=interpolation),
                transforms.CenterCrop(size=crop_size),
        ])
    elif "NoResize" in augment.keys():
        augment.pop("NoResize", None)
        pass
    else:
        transform_list.append(transforms.Resize(resolution))

    # has to be done at last
    random_erase = augment.pop("RandomErasing", {})
    random_erase_prob = random_erase.pop("p", 0.0)

    # add augmentations
    if isinstance(augment, dict):
        for key, value in augment.items():
            if "Own" in key:
                # random rotation does not allow for discrete choices
                transform_list.append(getattr(own_transforms, key)(**value))
            elif "StainColorJitterWrapper" in key:
                transform_list.append(getattr(staincolorjitter, key)(**value))
            elif "CIFAR10Policy" in key:
                transform_list.append(getattr(autoaugment, key)(**value))
            else:
                if "interpolation" in value:
                    if isinstance(value["interpolation"], str):
                        # use the specific interpolation mode
                        value["interpolation"] = getattr(InterpolationMode, value["interpolation"])
                    else:
                        # use the general interpolation mode
                        value["interpolation"] = getattr(InterpolationMode, interpolation)
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

    if random_erase_prob > 0:
        transform_list.append(transforms.RandomErasing(p=random_erase_prob))

    return transforms.Compose(transform_list)



def get_one_transform_old(
    resolution: int,
    augment: Union[bool, Dict],
    channel_wise_mean_images: List[float],
    channel_wise_std_images: List[float],
    validation: bool = False,
) -> transforms.Compose:
    """
    Standard transforms for images. Augmentations can be passed as a dictionary.
    """
    transform_list = []

    if isinstance(augment, bool):
        if augment:
            raise RuntimeError("Augmentations are not defined. Define them through a dictionary.")

        transform_list.extend([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

        # normalize
        add_normalize_transform(transform_list, channel_wise_mean_images, channel_wise_std_images)
        return transforms.Compose(transform_list)

    ###############################################
    # from here on, augment is a dictionary

    if isinstance(augment, DictConfig):
        augment = OmegaConf.to_container(
            augment, resolve=True, throw_on_missing=True
        )

    # which augmentations to use
    if validation:
        augment = {**augment.get("all", {}), **augment.get("eval", {})}
    else:
        augment = {**augment.get("all", {}), **augment.get("train", {})} 

    interpolation = getattr(InterpolationMode, augment.pop("interpolation", "BILINEAR"))


    # resize
    if "RandomResizedCrop" in augment.keys():
        kwargs = dict_for_transform(augment, "RandomResizedCrop")
        if "size" not in kwargs.keys():
            # if size is not given, use resolution
            kwargs["size"] = resolution
        transform_list.append(transforms.RandomResizedCrop(**kwargs, interpolation=interpolation))
    elif "short_side_center_crop" in augment.keys():
        kwargs = dict_for_transform(augment, "short_side_center_crop")
        # if size is not given, use resolution
        # like this we can resize and crop to different sizes like in the torchvision classification script
        resize_size = kwargs.get("resize_size", resolution)
        crop_size = kwargs.get("crop_size", resolution)
        transform_list.extend([
                transforms.Resize(size=resize_size, interpolation=interpolation),
                transforms.CenterCrop(size=crop_size),
        ])
    elif "NoResize" in augment.keys():
        dict_for_transform(augment, "NoResize")
    else:
        transform_list.append(transforms.Resize(resolution))

    # has to be done at last
    random_erase_kwargs = dict_for_transform(augment, "RandomErasing")
    random_erase_prob = random_erase_kwargs.pop("p", 0.0)

    # add augmentations
    if isinstance(augment, dict):
        for key, value in augment.items():
            if "Own" in key:
                # random rotation does not allow for discrete choices
                transform_list.append(getattr(own_transforms, key)(**value))
            elif "CIFAR10Policy" in key:
                transform_list.append(getattr(autoaugment, key)(**value))
            else:
                if "interpolation" in value:
                    if isinstance(value["interpolation"], str):
                        # use the specific interpolation mode
                        value["interpolation"] = getattr(InterpolationMode, value["interpolation"])
                    else:
                        # use the general interpolation mode
                        value["interpolation"] = getattr(InterpolationMode, interpolation)
                transform_list.append(getattr(transforms, key)(**value))
    elif isinstance(augment, bool):
        NotImplementedError("Bool Augmentation is not implemented yet")
    else:
        raise RuntimeError("Unknown Augmentation Type")

    transform_list.extend([transforms.ToTensor()])

    # normalize
    add_normalize_transform(transform_list, channel_wise_mean_images, channel_wise_std_images)


    if random_erase_prob > 0:
        transform_list.append(transforms.RandomErasing(p=random_erase_prob))

    return transforms.Compose(transform_list)


def add_normalize_transform(
    transform_list: List,
    channel_wise_mean_images: List[float],
    channel_wise_std_images: List[float],
) -> None:
    """
    Normalize images
    """
    if channel_wise_mean_images is not None and channel_wise_std_images is not None:
        transform_list.extend([
            transforms.Normalize(
                mean=channel_wise_mean_images,
                std=channel_wise_std_images,
            )
        ])


def dict_for_transform(augment: Dict, element_string: str) -> Dict:
    """
    Removes the element from the dictionary and returns a dict or raises and error.
    """

    if element_string not in augment.keys():
        return {}
    
    kwargs_element = augment.pop(element_string)

    if isinstance(kwargs_element, bool):
        # in this case we return an empty dictionary
        return {}
    elif isinstance(kwargs_element, dict):
        return kwargs_element
    else:
        raise RuntimeError(f"Unknown augmentation type for {element_string}")
    

