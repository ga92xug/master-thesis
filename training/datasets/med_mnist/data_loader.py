
import re
from sklearn.model_selection import train_test_split
from sympy import root
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
from omegaconf import DictConfig

import medmnist
from medmnist import INFO, Evaluator
#from medmnist import INFO, Evaluator

import sys
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
sys.path.append(f"{os.getcwd()}")
from training.datasets.utils import get_normalize_weights, get_transforms

from training.datasets.predefined.autoaugment import CIFAR10Policy, Cutout

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

def get_transforms_cifar(name, channel_wise_mean_images, channel_wise_std_images, augment=False, rotation=False):
    # define transforms
    normalize = transforms.Normalize(mean=channel_wise_mean_images, std=channel_wise_std_images)

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(16),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if rotation:
        train_transform.transforms.insert(0, transforms.RandomRotation((0,360), Image.BILINEAR))
        valid_transform.transforms.insert(0, transforms.RandomRotation((0,360), Image.BILINEAR))
    
    return train_transform, valid_transform

def custom_collate(batch):
    data, labels = zip(*batch)
    data = torch.stack(data)
    assert len(labels[0]) == 1, "Assuming label is wrapped as a single-element list"
    
    labels = torch.tensor(np.array(labels), dtype=torch.int64)
    labels = labels.squeeze()
    return data, labels

def build_loaders(
        batch_size,
        eval_batch_size,
        data_dir,
        name,
        channel_wise_mean_images,
        channel_wise_std_images,
        n_out_classes,
        perturbation_test = False,
        perturbation_location = None,
        reduction_factor=None,
        should_normalize_weights=True,
        workers=8,
        augment=False,
        reshuffle=True,
        cfg=None,
        **kwargs,
    ):
    
    location = data_dir + "medmnist/"

    DataClass = getattr(medmnist, name)

    # Define the transformations
    train_transform = get_transforms(28, augment, channel_wise_mean_images, channel_wise_std_images)
    valid_transform = get_transforms(28, False, channel_wise_mean_images, channel_wise_std_images)
    
    #train_transform, valid_transform = get_transforms(name, channel_wise_mean_images, channel_wise_std_images, augment, 0)
    
    if not os.path.exists(location):
        os.makedirs(location)
    train_dataset = DataClass(split='train', transform=train_transform, download=True, root=location)
    valid_dataset = DataClass(split='val', transform=valid_transform, download=True, root=location)
    test_dataset = DataClass(split='test', transform=valid_transform, download=True, root=location)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=custom_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=workers, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=workers, collate_fn=custom_collate)

    # normalize weights
    labels = train_dataset.labels.squeeze()
    normalized_weights = get_normalize_weights(labels) if should_normalize_weights else 1

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    
    return loaders, normalized_weights

