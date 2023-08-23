import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image
import hydra
from omegaconf import DictConfig
#import cv2

import h5py
import numpy as np

import sys
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
sys.path.append(f"{os.getcwd()}")
#print("current working directory: ", os.getcwd())
from experiment.datasets.utils import get_normalize_weights
#from experiment import dataloader


def split_without_stratify(images, labels, random_seed):
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

def split_with_stratify(images, labels, random_seed, train_val_sizes=None):
    if train_val_sizes is None:
        train_images, val_test_images, train_labels, val_test_labels = train_test_split(*[images, labels], test_size=0.20, random_state=random_seed, stratify=labels)
        test_images, val_images, test_labels, val_labels = train_test_split(*[val_test_images, val_test_labels] , test_size=0.5, random_state=random_seed, stratify=val_test_labels)
    else:
        max_value_label = max(labels)
        print("max_value_label: ", max_value_label, "set(labels): ", set(train_val_sizes))
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(*[images, labels], train_size=sum(train_val_sizes) * max_value_label, random_state=random_seed, stratify=labels)
        train_images, val_images, train_labels, val_labels = train_test_split(*[train_val_images, train_val_labels], train_size=0.6, test_size=0.4, random_state=random_seed, stratify=train_val_labels)
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

class Custom_Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
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

def build_loaders(
    images, 
    labels,
    transform,
    batch_size,
    eval_batch_size,
    workers,
    split_function,
    train_val_sizes=None,
):
    random_seed = 42
    # Split the data into train, val, and test arrays.
    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_function(images, labels, random_seed=random_seed, train_val_sizes=train_val_sizes)

    # Create the DataLoaders
    train_loader = DataLoader(Custom_Dataset(train_images, train_labels, transform=transform), batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(Custom_Dataset(val_images, val_labels, transform=transform), batch_size=eval_batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(Custom_Dataset(test_images, test_labels, transform=transform), batch_size=eval_batch_size, shuffle=False, num_workers=workers)

    dataloaders = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }

    return dataloaders


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
    **kwargs,
):
    location = data_dir + name + "/Galaxy10_DECals.h5"
    #print("location: ", location)
    with h5py.File(location, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    images = images.astype(np.uint8)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_wise_mean_images, std=channel_wise_std_images),
    ])

    # normalize weights
    if should_normalize_weights:
        normalized_weights = get_normalize_weights(labels)
    else:
        # to gather the weighted accuracy
        normalized_weights = 1

    dataloaders = build_loaders(images, labels, transform, batch_size, eval_batch_size, workers, split_without_stratify)
    return dataloaders, normalized_weights


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
    train_val_sizes=None,
    **kwargs,
):
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
    #images = df['name'].apply(lambda file_location: np.array(Image.open(file_location)).astype(np.uint8)).values
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_wise_mean_images, std=channel_wise_std_images),
    ])

    # normalize weights
    if should_normalize_weights:
        normalized_weights = get_normalize_weights(labels)
    else:
        # to gather the weighted accuracy
        normalized_weights = 1

    dataloaders = build_loaders(images, labels, transform, batch_size, eval_batch_size, workers, split_with_stratify, train_val_sizes=train_val_sizes)
    return dataloaders, normalized_weights


@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    #print(cfg)
    dataloaders, normalize_weights = hydra.utils.call(cfg.training.dataset)

    train_dataloader = dataloaders["train"]
    valid_dataloader = dataloaders["valid"]
    test_dataloader = dataloaders["test"]

    max_val = 0
    for i, (images, labels) in enumerate(train_dataloader):
        print(images.shape)
        print(labels)

        break
    

if __name__ == "__main__":
    main()