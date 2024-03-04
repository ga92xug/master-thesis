from os import path
from pathlib import Path
import pathlib
from typing import Callable, Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange
from pickle import load
import sys
import os
sys.path.append(f"{os.getcwd()}")
from src.data.datasets.custom.get_data import subject_split_image_label_folder

def train_data(
    folder:str,
    exclude_1_channel: bool = False,
) -> Tuple[List[str], List[int]]:
    images = []
    labels = []
    for label in os.listdir(folder):
        folder_label = path.join(folder, label)
        if not path.isdir(folder_label):
            # these are files like .DS_Store
            continue
        for i, image in enumerate(os.listdir(folder_label)):
            image_path = path.join(folder_label, image)
            
            
            images.append(image_path)
            labels.append(label)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels

def get_imagenet(
    data_dir: str,
    name: str,
    resolution: int,
) -> Tuple[Dict, Dict]:
    
    location = path.join(data_dir, name)

    train_loc = path.join(location, 'train') 
    val_loc = path.join(location, 'val')
    test_loc = path.join(location, 'test')

    train_images, train_labels = subject_split_image_label_folder(train_loc)
    val_images, val_labels = None, None # get_images_and_labels_from_folder(val_loc)
    test_images, test_labels = None, None # get_images_and_labels_from_folder(test_loc)

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

    return images, labels


if __name__ == "__main__":
    print("start")
    data_dir = "/home/atuin/b180dc/b180dc27/datasets/"
    name = "ILSVRC2012"
    images, labels = get_imagenet(data_dir=data_dir, name=name, resolution=0)
    print("loaded imagenet")

    train_images = images["train"]
    train_labels = labels["train"]

    print(train_images)
    for image, label in zip(train_images, train_labels):
        print(label, image)
        break


