
import os
import tarfile
import wget
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler

import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

ROOT_DIR = '../Data/frischs/datasets/'
DATA_DIR = ROOT_DIR + "imagenette/imagenette2-320/" 

# adpated from https://github.com/pytorch/TensorRT/blob/main/notebooks/qat-ptq-workflow.ipynb

# Copyright 2022 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


def download_data(DATA_DIR):
    if os.path.exists(DATA_DIR):
        # download full sized with 'imagenette2'
        if not os.path.exists(os.path.join(DATA_DIR, 'imagenette2-320')):
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
            wget.download(url)
            # open file
            file = tarfile.open('imagenette2-320.tgz')
            # extracting file
            file.extractall(DATA_DIR)
            file.close()
    else:
        print("This directory doesn't exist. Create the directory and run again")

def build_imagenette_loaders(batch_size,
                          eval_batchsize,
                          num_workers=8,
                          augment=False,
                          drop_last=False,
                          resolution=108,
                          #resolution_scaling=1.0,
                          resolution_test=True
                          ):
    image_size = resolution # int(160 * resolution_scaling)

    # download_data(DATA_DIR)
    # Define training and validation data paths
    if image_size > 320:
        DATA_DIR = ROOT_DIR + "imagenette/imagenette2/"
    elif image_size > 160:
        DATA_DIR = ROOT_DIR + "imagenette/imagenette2-320/"
    else:
        DATA_DIR = ROOT_DIR + "imagenette/imagenette2-160/"

    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VAL_DIR = os.path.join(DATA_DIR, 'val')

    #Performing Transformations on the dataset and defining training and validation dataloaders
    if image_size <= 256 and not resolution_test:
        image_size_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            ])
    elif image_size > 256 or resolution_test:
        image_size_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            ])

    valid_transform = transforms.Compose([
            image_size_transform,
            transforms.ToTensor(),
            ])

    if augment:
        train_transform = transforms.Compose([
                image_size_transform,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])
    else:
        train_transform = valid_transform

    

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=valid_transform)
    test_dataset = torch.utils.data.random_split(val_dataset, [0.7, 0.3])[1]

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, 
                                       num_workers=num_workers,
                                       shuffle=True, drop_last=drop_last)
    val_dataloader = data.DataLoader(val_dataset, batch_size=eval_batchsize, 
                                     num_workers=num_workers,
                                     shuffle=False, drop_last=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=eval_batchsize, 
                                      num_workers=num_workers,
                                      shuffle=False, drop_last=False)
    
    n_inputs = 3
    n_classes = 10
    
    return train_dataloader, val_dataloader, test_dataloader, n_inputs, n_classes


if __name__ == "__main__":
        train_dataloader, val_dataloader, test_dataloader, n_inputs, n_classes = build_imagenette_loaders(
            batch_size=128,
            eval_batchsize=128,
            num_workers=8,
            augment=False,
            resolution_scaling=3.0,
            resolution_test=True,
        )
        
        print(len(train_dataloader.dataset))
        print(len(val_dataloader.dataset))
        print(len(test_dataloader.dataset))
    
        print(n_inputs)
        print(n_classes)
    
        for i, (images, labels) in enumerate(train_dataloader):
            print(images.shape)
            print(labels.shape)
            break
            
    
        for i, (images, labels) in enumerate(val_dataloader):
            print(images.shape)
            print(labels.shape)
            break
            
    
        for i, (images, labels) in enumerate(test_dataloader):
            print(images.shape)
            print(labels.shape)
            break        
    
        print("Done")
    
        # print(train_loader.dataset[0][0].shape)
        # print(train_loader.dataset[0][1].shape)
        # print(train_loader.dataset[0][1])
    
        # print(valid_loader.dataset[0][0].shape)
        # print(valid_loader.dataset[0][1].shape)
        # print(valid_loader.dataset[0][1])
    
        # print(test_loader.dataset[0][0].shape)
        # print(test_loader.dataset[0][1].shape)
        # print(test_loader.dataset[0][1])
    
        # print(len(train_loader.dataset))
        # print(len(valid_loader.dataset))
        # print(len(test_loader.dataset))
    
        # print(n_inputs)
        # print(n_classes)
    
        # for i, (images, labels) in enumerate(train_loader):
        #     print(images.shape)
        #     print(labels.shape)
        #     break
    
        # for i, (images, labels) in enumerate(valid_loader):
        #     print(images.shape)
        #     print(labels.shape)
        #     break
    
        # for i, (images, labels) in enumerate(test_loader):
        #     print(images.shape)
        #     print(labels.shape)
        #     break
    
        # print("Done")
    
        # print(train_loader.dataset[0][0].shape)
        # print(train_loader.dataset[0][1].shape)
        # print(train_loader.dataset[0][1])
    
        # print(valid_loader.dataset[0][0].shape)
        # print(valid_loader.dataset[0][1].shape)
        # print(valid_loader.dataset
