from typing import Dict, List, Union
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image

import h5py
import numpy as np

import sys
import os
from src.data.datasets.general.datasets import create_datasets
sys.path.append(f"{os.getcwd()}")
os.environ['HYDRA_FULL_ERROR'] = '1'
from training.datasets.utils import (
    get_normalize_weights, 
    get_transforms, 
    split_with_stratify, 
)

from training.datasets.general.utils import get_images_and_labels_DeepDRiD, get_images_and_labels_nct


def get_Galaxy10_DECals(
    data_dir: str,
    name: str,
    resolution: int,
    should_normalize_weights: bool,
    channel_wise_mean_images: list,
    channel_wise_std_images: list,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    augment: bool,
    verbose: int,
    **kwargs,
):
    assert resolution <= 256, "The maximum resolution for Galaxy10_DECals is 256"

    location = data_dir + name + "/Galaxy10_DECals.h5"
    #print("location: ", location)
    with h5py.File(location, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    images = images.astype(np.uint8)

    return create_datasets(
        images=images, 
        labels=labels, 
        batch_size=batch_size, 
        eval_batch_size=eval_batch_size, 
        workers=workers, 
        reduction_factor=reduction_factor,
        val_size=0.1,
        test_size=0.1,
        verbose=verbose,
    )


def get_ISIC_2019(
    data_dir: str,
    name: str,
    resolution: int,
    should_normalize_weights: bool,
    channel_wise_mean_images: list,
    channel_wise_std_images: list,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    augment: bool,
    verbose: int,
    reduction_factor=None,
    **kwargs,
):
    assert resolution <= 450, \
        "The maximum resolution for ISIC_2019 is 450x450 since the minimum height is 450"

    location = data_dir + name
    # these files you download
    ground_truth = location + '/ISIC_2019_Training_GroundTruth.csv'
    images = location + '/ISIC_2019_Training_Input'

    df = pd.read_csv(ground_truth)
    for label in df.columns[1:]:
        df.loc[df[label] == 1.0, 'label'] = label
    
    #create instance of label encoder
    lab = LabelEncoder()
    df['label'] = lab.fit_transform(df['label'])
        
    df.rename(columns={'image': 'name'}, inplace=True)
    df['name'] = df['name'].apply(lambda x: "{}/{}.jpg".format(images,x))
    df = df[['name', 'label']]
    labels = df['label'].values
    images = df['name'].values
    
    return create_datasets(
        images=images, 
        labels=labels, 
        batch_size=batch_size, 
        eval_batch_size=eval_batch_size, 
        workers=workers, 
        reduction_factor=reduction_factor,
        val_size=0.1,
        test_size=0.1,
        verbose=verbose,
    )


def get_OCT(
        data_dir: str,
        name: str,
        resolution: int,
        should_normalize_weights: bool,
        channel_wise_mean_images: list,
        channel_wise_std_images: list,
        batch_size: int,
        eval_batch_size: int,
        workers: int,
        augment: bool,
        verbose: int,
        reduction_factor: float = 1,
        test_as_valid: bool = False,
        **kwargs,
    ):
    location = data_dir + name + "/CellData/OCT"

    train_images, train_labels = get_images_and_labels_nct(location + "/train/")
    test_images, test_labels = get_images_and_labels_nct(location + "/test/")

    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        original_augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images,
        verbose=verbose,
    )

    # normalize weights
    normalized_weights = get_normalize_weights(train_labels, verbose) if should_normalize_weights else 1

    images = {
        "train": train_images,
        "test": test_images,
    }
    labels = {
        "train": train_labels,
        "test": test_labels,
    }

    dataloaders = create_datasets(
        images=images, 
        labels=labels, 
        train_transform=train_transform, 
        valid_transform=valid_transform, 
        batch_size=batch_size, 
        eval_batch_size=eval_batch_size, 
        workers=workers, 
        reduction_factor=reduction_factor,
        val_size=0.1,
        test_size=0.0,
        test_as_valid=test_as_valid,
        verbose=verbose,
    )

    return dataloaders, normalized_weights


def get_nct(
        data_dir: str,
        name: str,
        resolution: int,
        should_normalize_weights: bool,
        channel_wise_mean_images: list,
        channel_wise_std_images: list,
        batch_size: int,
        eval_batch_size: int,
        workers: int,
        augment: bool,
        verbose: int,
        reduction_factor: float = 1,
        test_as_valid: bool = False,
        **kwargs,
    ):
    location = data_dir + name 

    train_images, train_labels = get_images_and_labels_nct(location + "/NCT-CRC-HE-100K/")
    test_images, test_labels = get_images_and_labels_nct(location + "/CRC-VAL-HE-7K/")
    print("train_images: ", train_images[:2])
    print("train_labels: ", train_labels[:2])
    print("test_images: ", test_images[:2])
    print("test_labels: ", test_labels[:2])

    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        original_augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images,
        verbose=verbose,
    )

    # normalize weights
    normalized_weights = get_normalize_weights(train_labels, verbose) if should_normalize_weights else 1

    images = {
        "train": train_images,
        "test": test_images,
    }
    labels = {
        "train": train_labels,
        "test": test_labels,
    }

    dataloaders = create_datasets(
        images=images, 
        labels=labels, 
        train_transform=train_transform, 
        valid_transform=valid_transform, 
        batch_size=batch_size, 
        eval_batch_size=eval_batch_size, 
        workers=workers, 
        reduction_factor=reduction_factor,
        val_size=0.1,
        test_size=0.0,
        test_as_valid=test_as_valid,
        verbose=verbose,
    )

    return dataloaders, normalized_weights


def get_blood(
        data_dir: str,
        name: str,
        resolution: int,
        should_normalize_weights: bool,
        channel_wise_mean_images: list,
        channel_wise_std_images: list,
        batch_size: int,
        eval_batch_size: int,
        workers: int,
        augment: bool,
        verbose: int,
        reduction_factor: float = 1,
        test_as_valid: bool = False,
        **kwargs,
    ):
    location = data_dir + name + "/PBC_dataset_normal_DIB/"

    images, labels = get_images_and_labels_nct(location)

    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        original_augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images,
        verbose=verbose,
    )

    # normalize weights
    normalized_weights = get_normalize_weights(labels, verbose) if should_normalize_weights else 1

    dataloaders = create_datasets(
        images=images, 
        labels=labels, 
        train_transform=train_transform, 
        valid_transform=valid_transform, 
        batch_size=batch_size, 
        eval_batch_size=eval_batch_size, 
        workers=workers, 
        reduction_factor=reduction_factor,
        val_size=0.1,
        test_size=0.2,
        test_as_valid=test_as_valid,
        verbose=verbose,
    )

    return dataloaders, normalized_weights


def get_DeepDRiD(
        data_dir: str,
        name: str,
        resolution: int,
        should_normalize_weights: bool,
        channel_wise_mean_images: list,
        channel_wise_std_images: list,
        batch_size: int,
        eval_batch_size: int,
        workers: int,
        augment: bool,
        mode: str,
        verbose: int,
        reduction_factor: float = 1,
        test_as_valid: bool = False,
        **kwargs,
    ):
    location = data_dir + "DeepDRiD/DeepDRiD-master/regular_fundus_images/"

    df_train = pd.read_csv(location + 'regular-fundus-training/regular-fundus-training.csv')
    df_val = pd.read_csv(location + 'regular-fundus-validation/regular-fundus-validation.csv')
    df_test = pd.read_excel(location + 'Online-Challenge1&2-Evaluation/Challenge2_labels.xlsx')

    train_images, train_labels = get_images_and_labels_DeepDRiD(df_train, location + "regular-fundus-training/", mode=mode)
    val_images, val_labels = get_images_and_labels_DeepDRiD(df_val, location + "regular-fundus-validation/", mode=mode)
    test_images, test_labels = get_images_and_labels_DeepDRiD(df_test, location + "Online-Challenge1&2-Evaluation/", mode=mode, test=True)

    # Define the transformations
    train_transform, valid_transform = get_transforms(
        resolution=resolution, 
        original_augment=augment, 
        channel_wise_mean_images=channel_wise_mean_images, 
        channel_wise_std_images=channel_wise_std_images,
        verbose=verbose,
    )

    # normalize weights
    normalized_weights = get_normalize_weights(train_labels, verbose) if should_normalize_weights else 1

    images = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }
    labels = {
        "train": train_labels,
        "val": val_labels,
        "test": test_labels,
    }

    dataloaders = create_datasets(
        images=images, 
        labels=labels, 
        train_transform=train_transform, 
        valid_transform=valid_transform, 
        batch_size=batch_size, 
        eval_batch_size=eval_batch_size, 
        workers=workers, 
        reduction_factor=reduction_factor,
        val_size=0.0,
        test_size=0.0,
        test_as_valid=test_as_valid,
        verbose=verbose,
    )

    return dataloaders, normalized_weights