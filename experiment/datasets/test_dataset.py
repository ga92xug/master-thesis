import hydra
from omegaconf import DictConfig
import numpy as np

import sys
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
sys.path.append(f"{os.getcwd()}")

import torch
from tqdm import tqdm
import h5py
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


MEAN = [0.01443392, 0.01443392, 0.01443392]
STD = [0.12014125, 0.1125077, 0.10281154]

class Galaxy10Dataset(torch.utils.data.Dataset):
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

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.int64)

        return image, label


def build_galaxy10_loaders(
        batch_size,
        eval_batch_size,
        data_dir,
        name,
        resolution,
        workers=8,
        augment=False,
    ):
    
    print("Galaxy10 dataset", "batch size", batch_size, "eval batch size", eval_batch_size)

    location = data_dir + name + "/Galaxy10_DECals.h5"

    with h5py.File(location, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    images = images.astype(np.uint8)

    # Split the data into train, val, and test arrays.
    np.random.seed(42)
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

    valid_transform = transforms.Compose([
                transforms.Resize(resolution),
                #transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    if augment:
        train_transform = transforms.Compose([
                transforms.Resize(resolution),
                #transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
    else:
        train_transform = valid_transform

    # Create the train, val, and test datasets.
    train_dataset = Galaxy10Dataset(train_images, train_labels, transform=train_transform)
    val_dataset = Galaxy10Dataset(val_images, val_labels, transform=valid_transform)
    test_dataset = Galaxy10Dataset(test_images, test_labels, transform=valid_transform)


    # Create the train, val, and test dataloaders.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=workers,
        )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=workers,
        )

    n_inputs = 3
    n_classes = 10
    
    dataloaders = {
        "train": train_dataloader,
        "valid": val_dataloader,
        "test": test_dataloader,
    }

    normalize_weight = 1

    return dataloaders, n_inputs, resolution, n_classes, normalize_weight


def regular_instantiate():
    dataloaders, n_inputs, resolution, n_classes, normalize_weight = build_galaxy10_loaders(batch_size=128, eval_batch_size=128, data_dir= "../../Data/frischs/datasets/" , name="Galaxy10_DECals", resolution=108, workers=8, augment=False)
    train_dataloader = dataloaders["train"]
    valid_dataloader = dataloaders["valid"]
    test_dataloader = dataloaders["test"]

    return dataloaders, n_inputs, resolution, n_classes, normalize_weight

"""
@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def hydra_dataloader_instantiate(cfg: DictConfig):
    #print(cfg)
    dataloaders, normalize_weights = hydra.utils.call(cfg.training.dataset)
    n_inputs = cfg.training.dataset.n_in_channels
    n_outputs = cfg.training.dataset.n_out_classes
    image_size = cfg.training.dataset.resolution

    train_dataloader = dataloaders["train"]
    valid_dataloader = dataloaders["valid"]
    test_dataloader = dataloaders["test"]

    return dataloaders, n_inputs, image_size, n_outputs, normalize_weights

    #max_val = 0
    #for i, (images, labels) in enumerate(train_dataloader):
    #    print(images.shape)
    #    #print(labels)

    #    break
"""
import torch
def compare_dataloader_outputs(output1, output2):
    if not isinstance(output1, torch.Tensor):
        output1 = torch.tensor(output1)
        output2 = torch.tensor(output2)

    if torch.equal(output1, output2):
        print("Data from dataloaders is the same.")
    else:
        print("Data from dataloaders is different.")

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def hydra_dataloader_instantiate(cfg: DictConfig):
    dataloaders_hydra, normalize_weights = hydra.utils.call(cfg.training.dataset)
    n_inputs_hydra = cfg.training.dataset.n_in_channels
    resolution_hydra = cfg.training.dataset.resolution
    n_classes_hydra = cfg.training.dataset.n_out_classes

    # Call regular instantiation to get its outputs
    dataloaders_reg, n_inputs_reg, resolution_reg, n_classes_reg, normalize_weight_reg = regular_instantiate()

    # Compare the first batch from the dataloaders
    train_dataloader_hydra = dataloaders_hydra["train"]
    train_dataloader_reg = dataloaders_reg["train"]
    
    images_hydra, labels_hydra = next(iter(train_dataloader_hydra))
    images_reg, labels_reg = next(iter(train_dataloader_reg))
    
    compare_dataloader_outputs(images_hydra, images_reg)
    compare_dataloader_outputs(labels_hydra, labels_reg)

    # Compare other relevant outputs
    compare_dataloader_outputs(n_inputs_hydra, n_inputs_reg)
    compare_dataloader_outputs(resolution_hydra, resolution_reg)
    compare_dataloader_outputs(n_classes_hydra, n_classes_reg)
    compare_dataloader_outputs(normalize_weights, normalize_weight_reg)

if __name__ == "__main__":
    hydra_dataloader_instantiate()
 