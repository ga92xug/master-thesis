from typing import Dict, List, Tuple, Union

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import h5py
import numpy as np
from os import path
from PIL import Image


def get_galaxy10(
    data_dir: str,
    name: str,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    assert resolution <= 256, "The maximum resolution for Galaxy10_DECals is 256"

    location = data_dir + name + "/Galaxy10_DECals.h5"
    #print("location: ", location)
    with h5py.File(location, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    images = images.astype(np.uint8)

    return images, labels

def get_imagenette(
    data_dir: str,
    name: str,
    resolution: int,
) -> Tuple[Dict, Dict]:
    
    # Define training and validation data paths
    if resolution > 320:
        resolution_folder = "imagenette2/"
    elif resolution > 160:
        resolution_folder = "imagenette2-320/"
    else:
        resolution_folder = "imagenette2-160/"

    location = path.join(data_dir, name, resolution_folder)

    train_loc = path.join(location, 'train') 
    val_loc = path.join(location, 'val')

    train_images, train_labels = get_images_and_labels_from_folder(train_loc, exclude_1_channel=True)
    val_images, val_labels = get_images_and_labels_from_folder(val_loc, exclude_1_channel=True)

    images = {
        "train": train_images,
        "test": val_images,
    }

    labels = {
        "train": train_labels,
        "test": val_labels,
    }    

    return images, labels


def get_ISIC_2019(
    data_dir: str,
    name: str,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
    
    return images, labels


def get_OCT(
    data_dir: str,
    name: str,
    resolution: int,
) -> Tuple[Dict, Dict]:
    location = data_dir + name + "/CellData/OCT"

    train_images, train_labels = get_images_and_labels_from_folder(location + "/train/")
    test_images, test_labels = get_images_and_labels_from_folder(location + "/test/")

    images = {
        "train": train_images,
        "test": test_images,
    }
    labels = {
        "train": train_labels,
        "test": test_labels,
    }

    return images, labels


def get_nct(
    data_dir: str,
    name: str,
    resolution: int,
) -> Tuple[Dict, Dict]:
    location = data_dir + name 

    train_images, train_labels = get_images_and_labels_from_folder(location + "/NCT-CRC-HE-100K/")
    test_images, test_labels = get_images_and_labels_from_folder(location + "/CRC-VAL-HE-7K/")

    images = {
        "train": train_images,
        "test": test_images,
    }
    labels = {
        "train": train_labels,
        "test": test_labels,
    }

    return images, labels


def get_blood(
    data_dir: str,
    name: str,
    resolution: int,
) -> Tuple[Dict, Dict]:
    location = data_dir + name + "/PBC_dataset_normal_DIB/"

    images, labels = get_images_and_labels_from_folder(location)

    return images, labels


def get_DeepDRiD(
    data_dir: str,
    name: str,
    resolution: int,
    mode: str,
) -> Tuple[Dict, Dict]:
    location = data_dir + "DeepDRiD/DeepDRiD-master/regular_fundus_images/"

    df_train = pd.read_csv(location + 'regular-fundus-training/regular-fundus-training.csv')
    df_val = pd.read_csv(location + 'regular-fundus-validation/regular-fundus-validation.csv')
    df_test = pd.read_excel(location + 'Online-Challenge1&2-Evaluation/Challenge2_labels.xlsx')

    train_images, train_labels = get_images_and_labels_DeepDRiD(df_train, location + "regular-fundus-training/", mode=mode)
    val_images, val_labels = get_images_and_labels_DeepDRiD(df_val, location + "regular-fundus-validation/", mode=mode)
    test_images, test_labels = get_images_and_labels_DeepDRiD(df_test, location + "Online-Challenge1&2-Evaluation/", mode=mode, test=True)

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

################################################################################
# Helper functions
################################################################################

def get_images_and_labels_DeepDRiD(df: pd.DataFrame, path: str, mode: str, test: bool = False):
    """
    Returns a list of image paths and a list of labels from a dataframe.
    The image path is transformed to fit the folder structure of the DeepDRiD dataset.
    """
    assert mode in ["Overall quality", "patient_DR_Level"]
    if not test:
        def replace_backslashes(input_string):
            # Split the string by backslashes and get everything after the second backslash
            parts = input_string.split("\\", 2)
            if len(parts) >= 3:
                result = parts[2].replace("\\", "/")
                return result
            else:
                raise ValueError("String does not contain enough backslashes.")

        images = df["image_path"].tolist()
        images = [path + "Images/" + replace_backslashes(image) for image in images]

    else:
        assert mode == "Overall quality", "Not implemented yet."

        images = df["image_id"].tolist()
        images = [path + "Images/" + image.split("_")[0] + "/" + image + ".jpg" for image in images]

    labels = df[mode].tolist()
    labels

    return images, labels


def get_images_and_labels_from_folder(
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
        for image in os.listdir(folder_label):
            image_path = path.join(folder_label, image)
            if exclude_1_channel:
                with Image.open(image_path) as img:
                    if len(np.array(img).shape) < 3 or np.array(img).shape[2] == 1:
                        # This is a grayscale image or has less than 3 dimensions, skip it
                        continue
                images.append(image_path)
                labels.append(label)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels