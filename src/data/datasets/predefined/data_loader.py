
import re
from sklearn.model_selection import train_test_split
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch

import sys
import os

os.environ['HYDRA_FULL_ERROR'] = '1'
sys.path.append(f"{os.getcwd()}")

from src.data.datasets.predefined.autoaugment import CIFAR10Policy, Cutout

MEAN = np.array([125.3, 123.0, 113.9]) / 255.0  # = np.array([0.49137255, 0.48235294, 0.44666667])
STD = np.array([63.0, 62.1, 66.7]) / 255.0  # = np.array([0.24705882, 0.24352941, 0.26156863])

class CIFAR_C(Dataset):
    def __init__(self, location, transform=None):
        self.transform = transform
        # CIFAR10 test images under different perturbations
        self.files = os.listdir(location)
        self.files = [f.split(".")[0] for f in self.files if "labels" not in f]
        
        self.images_perturbation_list = []
        self.labels_perturbations_list = []
        self.meta_data_list = []
        for i, file in enumerate(self.files):
            self.images_perturbation_list.append(np.load(location + file + '.npy'))
            self.labels_perturbations_list.append(np.load(location + 'labels.npy'))
            self.meta_data_list.append([i] * len(self.labels_perturbations_list[-1]))
        
        self.images_perturbation_list = np.concatenate(self.images_perturbation_list)
        self.labels_perturbations_list = np.concatenate(self.labels_perturbations_list)
        self.meta_data_list = np.concatenate(self.meta_data_list)
        
    def __len__(self):
        return len(self.images_perturbation_list)
    
    def __getitem__(self, index):
        image = self.images_perturbation_list[index]
        label = self.labels_perturbations_list[index]
        meta_data = self.meta_data_list[index]
        
        image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.int64)
        meta_data = torch.tensor(meta_data, dtype=torch.int64)
        
        return image, label, meta_data
    
    def eval(self, predictions, labels, meta_data_list):
        # predictions: (N,)
        # labels: (N,)
        # meta_data_list: (N,)
        # returns: (num_perturbations, num_classes)
        assert len(predictions) == len(labels) == len(meta_data_list)
        
        perturbations = np.unique(meta_data_list)
        
        # compute accuracy for each perturbation
        accs = {}
        for perturbation in perturbations:
            idx = meta_data_list == perturbation
            print("idx.shape: ", idx.shape)
            accs[f"acc_{self.files[perturbation]}"] = (predictions[idx] == labels[idx]).mean()

        # compute average accuracy over all perturbations
        accs["mCE"] = np.mean(list(accs.values()))

        return accs


def stratified_subset_indices(dataset, reduction_factor, num_classes, random_seed=42):
    num_train = len(dataset)
    total_images = num_train * reduction_factor
    total_images_per_class = np.round(total_images / num_classes).astype(int)

    assert total_images_per_class >= 50, "total_images_per_class must be at least 50."
    assert reduction_factor < 1.0 and reduction_factor > 0.0, "reduction_size must be between 0.0 and 1.0."

    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_indices_all = []
    val_indices_all = []
    train_images_per_class = int(round(total_images_per_class * 0.6))
    val_images_per_class = total_images_per_class - train_images_per_class
    for class_label, indices in class_indices.items():
        train_indices_one_class, val_indices_one_class = train_test_split(indices, train_size=train_images_per_class, test_size=val_images_per_class, random_state=random_seed)
        train_indices_all.extend(train_indices_one_class)
        val_indices_all.extend(val_indices_one_class)
    
    assert len(train_indices_all) == len(set(train_indices_all)), "Duplicate indices in train set."
    assert len(val_indices_all) == len(set(val_indices_all)), "Duplicate indices in validation set."
    assert len(set(train_indices_all).intersection(val_indices_all)) == 0, "Overlap between train and validation indices."

    print("train_size: ", len(train_indices_all))
    print("val_size: ", len(val_indices_all))
    print("total_size: ", len(train_indices_all) + len(val_indices_all))

    return train_indices_all, val_indices_all

def get_transforms(name, channel_wise_mean_images, channel_wise_std_images, augment=False, rotation=False):
    # define transforms
    normalize = transforms.Normalize(mean=channel_wise_mean_images, std=channel_wise_std_images)

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if "cifar" in name:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(16),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    elif name == "stl10":
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(96, padding=12),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Cutout(32),
                Cutout(60),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                # Cutout(24),
                Cutout(48),
                normalize,
            ])
        
    if rotation:
        train_transform.transforms.insert(0, transforms.RandomRotation((0,360), Image.BILINEAR))
        valid_transform.transforms.insert(0, transforms.RandomRotation((0,360), Image.BILINEAR))
    
    return train_transform, valid_transform


def build_loaders(
        batch_size,
        eval_batch_size,
        data_dir,
        name,
        channel_wise_mean_images,
        channel_wise_std_images,
        num_classes,
        perturbation_test = False,
        perturbation_location = None,
        reduction_factor=None,
        workers=8,
        augment=False,
        reshuffle=True,
        **kwargs,
    ):
    # the rotated cifar datasets are named "cifar10_rot" and "cifar100_rot"
    if len(name.split("_")) == 2:
        name, _ = name.split("_")
        rotation = True
    elif len(name.split("_")) == 1:
        name = name
        rotation = False
    else:
        raise ValueError("Unknown dataset name.")
    
    assert name in ["cifar10", "cifar100", "stl10"], "Unknown dataset name."
    assert "cifar" in name if perturbation_test == True else True, "Perturbation test is only available for cifar datasets."

    location = data_dir + name + "/"
    
    train_transform, valid_transform = get_transforms(name, channel_wise_mean_images, channel_wise_std_images, augment, rotation)
    
    # load the dataset
    if "cifar" in name:
        if name == "cifar10":
            dataset_class = datasets.CIFAR10
        elif name == "cifar100":
            dataset_class = datasets.CIFAR100
        
        train_dataset = dataset_class(root=location, train=True, download=False, transform=train_transform)
        valid_dataset = dataset_class(root=location, train=True, download=False, transform=valid_transform)
        if perturbation_test:
            location + perturbation_location
            assert os.path.exists(location + perturbation_location), "Perturbation location does not exist."
            test_dataset = CIFAR_C(location + perturbation_location, transform=valid_transform)
        else:
            test_dataset = dataset_class(root=location, train=False, download=False, transform=valid_transform)

    elif name == "stl10":
        train_dataset = datasets.STL10(root=location, split="train", download=False, transform=train_transform)
        valid_dataset = datasets.STL10(root=location, split="train", download=False, transform=valid_transform)
        test_dataset = datasets.STL10(root=location, split="test", download=False, transform=valid_transform)
    else:
        raise ValueError("Unknown dataset name.")

    if reduction_factor is None or reduction_factor == 1.0:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))

        if reshuffle:
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
    else:
        train_idx, valid_idx = stratified_subset_indices(train_dataset, reduction_factor=reduction_factor, num_classes=num_classes, random_seed=42)

    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    print("len(train_sampler): ", len(train_sampler))
    print("len(valid_sampler): ", len(valid_sampler))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=workers, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=eval_batch_size, sampler=valid_sampler,
        num_workers=workers, pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    
    normalized_weights = None
    return loaders, normalized_weights


if __name__ == "__main__":
        loaders, _ = build_loaders(
            batch_size=128,
            eval_batch_size=128,
            data_dir='../../Data/frischs/datasets/',
            name="cifar10",
            channel_wise_mean_images=MEAN,
            channel_wise_std_images=STD,
            perturbation_test = False,
            perturbation_location = None,
            train_val_sizes=[30,20],
            workers=8,
            augment=False,
            reshuffle=True,
        )

        train_loader = loaders["train"]
        valid_loader = loaders["valid"]
        test_loader = loaders["test"]
        
        print(len(train_loader))
        print(len(valid_loader))
        print(len(test_loader.dataset))
    
    
        for i, (images, labels) in enumerate(train_loader):
            print(images.shape)
            print(labels.shape)
            break
    
        for i, (images, labels) in enumerate(valid_loader):
            print(images.shape)
            print(labels.shape)
            break
    
        for i, (images, labels) in enumerate(test_loader):
            print(images.shape)
            print(labels.shape)
            break
    
        print("Done")
    

"""
@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    #print(cfg)
    dataloaders, normalize_weights = hydra.utils.call(cfg.training.dataset)
"""