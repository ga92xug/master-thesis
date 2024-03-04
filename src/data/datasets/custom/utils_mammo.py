from typing import Dict, List, Tuple, Union
import rootutils
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import h5py
import numpy as np
from os import path
from PIL import Image
import re

def subject_split_image_label_folder(
    folder:str,
    subject_func: callable = lambda x: x.split('.')[0],
    splits: List[float] = [],
    train_func: callable = None,
) -> List[Tuple[List[str], List[int]]]:
    images = {}
    labels = {}

    for i, label in enumerate(os.listdir(folder)):
        folder_label = path.join(folder, label)
        if not path.isdir(folder_label):
            # these are files like .DS_Store
            continue

        subjects = [subject_func(s) for s in os.listdir(folder_label)]
        subjects = list(set(subjects))

        if len(splits) == 0:
            subjects_dict = {0: subjects}
        else:
            subjects_dict = {0: subjects}
            for split_label, split in enumerate(splits):
                split_0, split_1 = train_test_split(
                    subjects_dict[split_label], train_size=split, random_state=42)
                subjects_dict[split_label] = split_0
                subjects_dict[split_label + 1] = split_1
            
        for split_label, subjects in subjects_dict.items():
            current_images, current_labels = gather_image_per_subject(
                folder=folder_label, 
                subjects=subjects, 
                subject_func=subject_func, 
                i=i,
                train_func=train_func,
            )
            if split_label in images:
                images[split_label].extend(current_images)
                labels[split_label].extend(current_labels)
            else:
                images[split_label] = current_images
                labels[split_label] = current_labels

    return images, labels


def gather_image_per_subject(
    folder:str,
    subjects: List[str],
    subject_func: callable,
    i: int,
    train_func: callable = None,
) -> List[Tuple[List[str], List[int]]]:
    current_images = []
    current_labels = []
    for subject in subjects:
        for image in os.listdir(folder):
            if train_func is not None:
                if train_func(image) != True:
                    continue
            else:
                if subject_func(image) != subject:
                    continue

            image_path = path.join(folder, image)
            current_images.append(image_path)
            current_labels.append(i)

    return current_images, current_labels
    

if __name__ == "__main__":
    data_dir = "/home/frischs/Data/frischs/datasets/mammography/combined/"
    name = ""
    resolution = 224
    images, labels = get_mammo(data_dir, name, resolution)
    print("images: ", images["train"][0])
    print("labels: ", labels["train"][0])
    print("len(images['train']): ", len(images['train']))
    print("len(images['val']): ", len(images['val']))
    print("len(images['test']): ", len(images['test']))
    
    print("len(labels['train']): ", len(labels['train']))
    print("len(labels['val']): ", len(labels['val']))
    print("len(labels['test']): ", len(labels['test']))