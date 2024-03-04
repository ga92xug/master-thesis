from typing import Tuple

from tqdm import tqdm
import cv2
import numpy as np

import numpy as np
import cv2
from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch
import os
import imghdr
import torch
#import lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from glob import glob
from torchvision.io import read_image
import cv2 
from collections import Counter
from PIL import Image
import sys
sys.path.append(os.getcwd())
from pre_processing.ddsm.ddsm import get_ddsm_df


PATH = os.path.expanduser("~/Data/frischs/datasets/mammo/")

def resize_clahe(
        img: np.ndarray, 
        clipLimit: int = 5, 
        resize: Tuple[int, int] = (224, 224)
    ) -> Tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=clipLimit)
    clahe_img = clahe.apply(img)

    return cv2.resize(img, resize), cv2.resize(clahe_img, resize)

def save_image(
        img: np.ndarray, 
        path: str
    ) -> None:
    # save as numpy array
    np.save(path, img)

def pre_process_INbreast(
        clipLimit: int = 5, 
        resize: Tuple[int, int] = (224, 224) 
    ):
    root = os.path.join(PATH, "INbreast")
    final_dataset = os.path.join(root, "np_dataset")
    df = pd.read_excel(os.path.join(root, "INbreast.xls"), skipfooter=2)

    df.columns = df.columns.str.capitalize()
    paths = glob(f"{root}/ALL-IMGS/*.dcm")
    df['path'] = df['File name'].apply(lambda x: [path for path in paths if path.split('/')[-1].split('_')[0] == str(x)][0])
    df['Lesion annotation status'].fillna('cancer', inplace=True)
    df['Lesion annotation status'] = df['Lesion annotation status'].str.upper()
    df['Lesion annotation status'] = df['Lesion annotation status'].apply(lambda x: "benign" if x == 'NO ANNOTATION (NORMAL)' else "malignant")
    #df['label'] = df['Lesion annotation status'].apply(lambda x: 0 if x == 'NORMAL' else 1)

    # create a folder based on the cancer status
    for status in df['Lesion annotation status'].unique():
        os.makedirs(os.path.join(final_dataset, status), exist_ok=True)

    for i, row in df.iterrows():
        numpy_img = np.array(pydicom.dcmread(row['path']).pixel_array)
        img, clahe_img = resize_clahe(numpy_img)
        save_image(img, os.path.join(final_dataset, row['Lesion annotation status'], f"{row['File']}.png"))


def create_save_location(
        path: str, 
        labels: list
    ) -> None:
    for label in labels:
        os.makedirs(os.path.join(path, label), exist_ok=True)


def process_save_DDSM(row, save_location_mode, resize):
    # open and process image
    img = Image.open(row['image_file_path'])
    numpy_img = np.array(img)
    img, clahe_img = resize_clahe(numpy_img, resize=resize)

    # save images
    image_location = os.path.join(save_location_mode, row['label'])
    img_id = row['image_file_path'].split('/')[-2:] 
    img_id = "_".join(img_id).replace(".jpg", "")
    # resized
    resized_image_location = os.path.join(image_location, f"resized{resize[0]}_{img_id}")
    np.save(resized_image_location, img)
    # clahe
    clahe_image_location = os.path.join(image_location, f"clahe{resize[0]}_{img_id}")
    np.save(clahe_image_location, clahe_img)

def pre_process_DDSM(
        resize: Tuple[int] = (224, 224),
    ):
    print("Preprocessing DDSM")
    ddsm_path = os.path.join(PATH, "CBIS-DDSM")
    save_location = os.path.join(ddsm_path, "np_dataset")
    datasets = get_ddsm_df(ddsm_path)
    os.makedirs(save_location, exist_ok=True)
    
    for mode, df in datasets.items():
        print(f"Processing {mode}")
        save_location_mode = os.path.join(save_location, mode)
        create_save_location(save_location_mode, df['label'].unique())
        for i, row in tqdm(df.iterrows()):
            process_save_DDSM(row, save_location_mode, resize)
            
    print("Preprocessing done")

if __name__ == "__main__":
    #pre_process_INbreast()
    pre_process_DDSM()