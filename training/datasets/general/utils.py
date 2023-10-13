import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

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

    print("Image ", images[0])

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



def plot_stacked_bar_chart(df: DataFrame, title: str = "Patient DR Level vs Overall Quality"):
    # Count the occurrences of each pair (Overall quality, patient_DR_Level)
    grouped_data = df[['Overall quality', 'patient_DR_Level']].groupby("patient_DR_Level").value_counts().unstack().fillna(0)
    
    # Extract values for each quality level
    quality_0 = grouped_data.loc[:, 0].values
    quality_1 = grouped_data.loc[:, 1].values

    # Create the bar chart
    barWidth = 0.35
    fig, ax = plt.subplots()

    # Set position of bar on X axis
    r1 = np.arange(len(quality_0))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, quality_0, color='b', width=barWidth, edgecolor='white', label='Overall Quality 0')
    plt.bar(r2, quality_1, color='r', width=barWidth, edgecolor='white', label='Overall Quality 1')

    # Add labels, title, and axes ticks
    plt.xlabel('Patient DR Level', fontweight='bold')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks([r + barWidth / 2 for r in range(len(quality_0))], grouped_data.index.tolist())

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
    return fig, ax


import matplotlib.pyplot as plt

def plot_multiple_bar_charts(dfs: list, titles: list):
    n = len(dfs)
    fig, axs = plt.subplots(1, n, figsize=(15, 6))

    for i in range(n):
        df = dfs[i]
        title = titles[i]
        ax = axs[i]

        # Count occurrences for the chart
        grouped_data = df[['Overall quality', 'patient_DR_Level']].groupby("patient_DR_Level").value_counts().unstack().fillna(0)
        
        # Extract values for each quality level
        quality_0 = grouped_data.loc[:, 0].values
        quality_1 = grouped_data.loc[:, 1].values

        # Bar width
        barWidth = 0.35

        # Set position of bar on X axis
        r1 = np.arange(len(quality_0))
        r2 = [x + barWidth for x in r1]

        # Make the plot
        ax.bar(r1, quality_0, color='b', width=barWidth, edgecolor='white', label='Overall Quality 0')
        ax.bar(r2, quality_1, color='r', width=barWidth, edgecolor='white', label='Overall Quality 1')

        # Add labels, title, and axes ticks
        ax.set_xlabel('Patient DR Level', fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xticks([r + barWidth / 2 for r in range(len(quality_0))])
        ax.set_xticklabels(grouped_data.index.tolist())

        # Add legend
        ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

