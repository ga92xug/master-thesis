
import os
import tarfile
import wget
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler



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


def build_loaders(
        batch_size,
        eval_batch_size,
        data_dir,
        name,
        channel_wise_mean_images,
        channel_wise_std_images,
        workers=8,
        augment=False,
        resolution=108,
        **kwargs,
    ):

    # download_data(DATA_DIR)
    # Define training and validation data paths
    if resolution > 320:
        resolution_folder = "/imagenette2/"
    elif resolution > 160:
        resolution_folder = "/imagenette2-320/"
    else:
        resolution_folder = "/imagenette2-160/"

    DATA_DIR = data_dir + name + resolution_folder

    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VAL_DIR = os.path.join(DATA_DIR, 'val')

    normalize = transforms.Normalize(mean=channel_wise_mean_images, std=channel_wise_std_images)
    valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            #normalize,
            ])

    if augment:
        train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #normalize,
                ])
    else:
        train_transform = valid_transform

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=valid_transform)
    test_dataset = random_split(val_dataset, [0.7, 0.3])[1]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                       num_workers=workers, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, 
                                     num_workers=workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, 
                                      num_workers=workers, shuffle=False)
    
    loaders = {
        "train": train_dataloader,
        "valid": valid_dataloader,
        "test": test_dataloader,
    }

    normalized_weights = None
    return loaders, normalized_weights


if __name__ == "__main__":
        loaders, _ = build_loaders(
            batch_size=9469,
            eval_batch_size=3925,
            data_dir='../../Data/frischs/datasets/',
            name="imagenette",
            channel_wise_mean_images=0,
            channel_wise_std_images=0,
            workers=8,
            augment=False,
        )

        train_loader = loaders["train"]
        valid_loader = loaders["valid"]
        test_loader = loaders["test"]
        
        print(len(train_loader.dataset))
        print(len(valid_loader.dataset))
        print(len(test_loader.dataset))
    
    
        for i, (images, labels) in enumerate(train_loader):
            print(images.shape)
            print(labels.shape)
            print("mean", np.mean(images.numpy(), axis=(0, 2, 3)), "std", np.std(images.numpy(), axis=(0, 2, 3)))
            # mean [0.46947703 0.4485053  0.41963148] std [0.26470277 0.25778076 0.27517352]
            break
    
        for i, (images, labels) in enumerate(valid_loader):
            print(images.shape)
            print(labels.shape)
            #print("mean", np.mean(images.numpy(), axis=(0, 2, 3)), "std", np.std(images.numpy(), axis=(0, 2, 3)))
            # mean [0.4686638  0.44649398 0.41711974] std [0.26641014 0.25737774 0.27325   ]
            break
    
        for i, (images, labels) in enumerate(test_loader):
            print(images.shape)
            print(labels.shape)
            print("mean", np.mean(images.numpy(), axis=(0, 2, 3)), "std", np.std(images.numpy(), axis=(0, 2, 3)))
            # mean [0.4655748  0.44496107 0.41673166] std [0.26885194 0.25771102 0.27451214]
            break
    
        print("Done")
    