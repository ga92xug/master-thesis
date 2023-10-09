import dis
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds import get_dataset

from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split

import sys
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
sys.path.append(f"{os.getcwd()}")
from training.datasets.utils import get_normalize_weights, get_transforms

def subset_sampler(train_data):
    # train_subset_sampler, val_subset_sampler = subset_sampler(train_data)
    total_size = len(train_data)

    # Define the ratio for the split (e.g., 80% training, 20% validation)
    train_ratio = 0.8
    val_ratio = 0.2

    # Calculate the sizes of the training and validation subsets
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Create a list of indices to split the DataLoader
    indices = list(range(total_size))

    # Randomly split indices for the training and validation subsets
    train_indices, val_indices = train_test_split(indices, train_size=train_size, test_size=val_size)

    assert set(train_indices).intersection(set(val_indices)) == set(), "train and val indices should not overlap"

    return SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)



def get_loaders(
        data_dir: str,
        name: str,
        resolution: int,
        should_normalize_weights: bool,
        channel_wise_mean_images: list,
        channel_wise_std_images: list,
        batch_size: int,
        eval_batch_size: int,
        workers: int,
        augment: bool,
        distribution_shift: bool,
        **kwargs,
    ):
    location = data_dir + name
    dataset = get_dataset(name, root_dir=location, download=False)

    # Define the transformations
    train_transform = get_transforms(resolution, augment, channel_wise_mean_images, channel_wise_std_images)
    valid_transform = get_transforms(resolution, False, channel_wise_mean_images, channel_wise_std_images)

    

    train_data = dataset.get_subset("train",transform=train_transform)  
    # there is an indistribution validation set if we don't want to use the distribution shift
    val_name = "val" if distribution_shift else "id_val"
    val_data = dataset.get_subset(val_name,transform=valid_transform)
    test_data = dataset.get_subset("test",transform=valid_transform)
    
    # normalize weights
    normalized_weights = get_normalize_weights(train_data.dataset._y_array) if should_normalize_weights else None

    # shuffle the train data is automatically done in the wilds dataloader
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size, num_workers=workers)
    val_loader = get_eval_loader("standard", val_data, batch_size=eval_batch_size, num_workers=workers)
    test_loader = get_eval_loader("standard", test_data, batch_size=eval_batch_size, num_workers=workers)

    dataloaders = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }

    return dataloaders, normalized_weights

