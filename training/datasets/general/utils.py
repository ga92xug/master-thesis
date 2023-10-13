import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def get_images_and_labels_DeepDRiD(df: pd.DataFrame, path: str):
    """
    Returns a list of image paths and a list of labels from a dataframe.
    The image path is transformed to fit the folder structure of the DeepDRiD dataset.
    """
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
    labels = df["Overall quality"].tolist()
    labels
    return images, labels


def get_images_and_labels_nct(folder:str):
    images = []
    labels = []
    for label in os.listdir(folder):
        for image in os.listdir(folder + label):
            images.append(folder + label + "/" + image)
            labels.append(label)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels