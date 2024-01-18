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
from src.data.datasets._transforms import own_transforms, autoaugment

def get_normalize_weights(
        labels: Union[List, np.ndarray],
        verbose: int,
    ) -> torch.Tensor:
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

    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


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
    if isinstance(reduction_factor, int):
        assert reduction_factor > 1, "reduction_factor should be greater than 1"
    else:
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
    if not reduction_factor == 1.0:
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
                stratify=labels if reduction_factor <= 1.0 else None 
                # stratify might not be feasible if we overfit (e.g. 10 images in train set)
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