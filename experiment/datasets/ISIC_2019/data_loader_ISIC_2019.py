import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image

import sys
import os
sys.path.append(f"{os.getcwd()}")
from experiment.datasets.utils import get_normalize_weights


def get_isic_df(
    dir: str,
    name: str,
):
    location = dir + name
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
    return df

class ISICDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx, 0])
        label = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    

def build_isic2019_loaders(
    batch_size,
    eval_batch_size,
    data_dir,
    name,
    resolution,
    workers,
    should_normalize_weights,
    augment=False,
):
    # Load the dataframe
    df = get_isic_df(data_dir, name)

    # normalize weights
    if should_normalize_weights:
        normalized_weights = get_normalize_weights(df)
    else:
        # to gather the weighted accuracy
        normalized_weights = 1

    # Split the data
    train_df, val_test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
    test_df, val_df = train_test_split(val_test_df, test_size=0.5, random_state=42, stratify=val_test_df['label'])

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6679, 0.5299, 0.5245], std=[0.1332, 0.1475, 0.1588]),
    ])

    # Create the DataLoaders
    train_loader = DataLoader(ISICDataset(train_df, transform=transform), batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(ISICDataset(val_df, transform=transform), batch_size=eval_batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(ISICDataset(test_df, transform=transform), batch_size=eval_batch_size, shuffle=False, num_workers=workers)

    n_inputs = 3
    n_classes = 8

    dataloaders = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }

    return dataloaders, n_inputs, resolution, n_classes, normalized_weights



if __name__ == '__main__':
    _dataloaders, n_inputs, image_size, n_outputs, normalize_weights = build_isic2019_loaders(batch_size=1024, eval_batch_size=16, data_dir= "../../Data/frischs/datasets/", name="ISIC_2019", resolution=224, workers=8, augment=False, should_normalize_weights=True)
    train_dataloader = _dataloaders["train"]
    valid_dataloader = _dataloaders["valid"]
    test_dataloader = _dataloaders["test"]

    max_val = 0
    for i, (images, labels) in enumerate(train_dataloader):
        print(images.shape)
        print(labels)

        break

