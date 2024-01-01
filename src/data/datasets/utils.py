from copy import deepcopy
from typing import List, Tuple
import numpy as np
import pandas as pd
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(f"{os.getcwd()}")
os.environ['HYDRA_FULL_ERROR'] = '1'
from training.datasets import own_transforms

def get_normalize_weights(
        labels: List or np.ndarray,
        verbose: int,
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
    if verbose > 2:
        print("weights: ", weights.tolist())
    return weights


def get_transforms(
        resolution: int,
        original_augment: bool or dict,
        channel_wise_mean_images: list,
        channel_wise_std_images: list,
        verbose: int ,
    ) -> Tuple[transforms.Compose, transforms.Compose]:

    augment = deepcopy(original_augment)
    train_transform = get_one_transform(resolution, augment, channel_wise_mean_images, channel_wise_std_images, validation=False)
    augment = deepcopy(original_augment)
    valid_transform = get_one_transform(resolution, augment, channel_wise_mean_images, channel_wise_std_images, validation=True)
    
    if verbose > 1:
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
    else:
        transform_list.append(transforms.Resize(resolution))

    # pop all size augmentations
    augment.pop("RandomResizedCrop", None)
    augment.pop("short_side_center_crop", None)

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

def split_with_stratify(
        images, 
        labels, 
        val_size: float,
        test_size: float,
        reduction_factor: float = 1.0,
    ):
    """
    Splits the images and labels into train, val, and test arrays.
    The train set can be reduced by the reduction_factor.

    Note: The test set is always the same regardless of the random seed set in the experiment.
        The other sets are split based on the numpy random seed.
    """

    assert val_size + test_size < 1.0, "val_size + test_size should be less than to 1.0"
    assert val_size >= 0.0, "val_size should be greater than or equal to 0.0"
    assert test_size >= 0.0, "test_size should be greater than or equal to 0.0"
    assert reduction_factor <= 1.0 and reduction_factor > 0.0, "reduction_factor should be in the range (0.0, 1.0]"
    
    if test_size > 0.0:
        images, test_images, labels, test_labels = \
            train_test_split(
                *[images, labels], 
                test_size=test_size, 
                random_state=42, # the test set should be the same for all experiments
                stratify=labels
            )
    else:
        test_images = None
        test_labels = None

    # Reduction
    if reduction_factor < 1.0:
        images, additional_images, labels, additional_labels = \
            train_test_split(
                *[images, labels], 
                train_size=reduction_factor, 
                stratify=labels
            )

    # Val_test
    if val_size > 0.0:
        train_images, val_images, train_labels, val_labels = \
            train_test_split(
                *[images, labels], 
                test_size=val_size, 
                stratify=labels
            )
    else:
        train_images = images
        train_labels = labels
        val_images = None
        val_labels = None
    
    
    # if reduction_factor < 1.0:
    #     # what to do with the additional images
    #     if test_size != 0.0:
    #         # if the test set is not distinct from the train set we can use the additional images for evaluation
    #         # otherwise no it breaks the independence of the test set
    #         test_images = np.concatenate([test_images, additional_images])
    #         test_labels = np.concatenate([test_labels, additional_labels])
    #     else:
    #         # we do not use the additional images for evaluation
    #         # we can not add them to the val set because that breaks the logic of reducing the size of the train set
    #         print("Not using additional images for evaluation.")

    return train_images, train_labels, val_images, val_labels, test_images, test_labels