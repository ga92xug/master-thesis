from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from lightning import LightningDataModule
from omegaconf import DictConfig, open_dict

import numpy as np
import hashlib
import torch

import pytest
import torch

import rootutils

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

from src.data.datamodule import DataModule
from tests.train.helpers.hydra_init import hydra_compose

DATASETS = ["cifar10", "isic2019", "mnist", "DeepDRiD", "blood", "imagenette"]

@pytest.mark.parametrize("dataset", DATASETS)
def test_datamodule(
    dataset: str,
    get_mean_std: bool = False,
    no_augment: bool = False,
) -> None:
    """Tests `DataModule` to verify that all the datasets can be loaded correctly.
      It checks that the necessary attributes were created 
      (e.g., the dataloader objects), and that dtypes and batch sizes 
      correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    

    overrides = [
        "training_setup/dataset=" + dataset,
    ]

    if no_augment:
        overrides.append("training_setup.dataset.augment=false")
    
    if get_mean_std:
        overrides.append("training_setup.dataset.channel_wise_mean_images=null")
        overrides.append("training_setup.dataset.channel_wise_std_images=null")
        overrides.append("training_setup.dataset.augment=false")

    cfg = hydra_compose(overrides)
    np.random.seed(cfg.seed)

    dm: LightningDataModule = DataModule(cfg.training_setup.dataset)
    dm.prepare_data()

    assert not dm.train_set and not dm.val_set and not dm.test_set

    dm.setup()
    assert dm.train_set and dm.val_set and dm.test_set
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    len_train = len(dm.train_set)
    len_val = len(dm.val_set)
    len_test = len(dm.test_set)
    num_datapoints = len_train + len_val + len_test
    print("len_train: ", len_train, "len_val: ", len_val, "len_test: ", len_test)
    print("num_datapoints: ", num_datapoints)

    dataloaders = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }

    label_counts_list_modes = []
    for mode, dataloader in dataloaders.items():
        batch = next(iter(dataloader))
        x, y = batch
        if mode == "train":
            batch_size = cfg.training_setup.dataset.batch_size
        else:
            batch_size = cfg.training_setup.dataset.eval_batch_size

        assert len(x) == batch_size
        assert len(y) == batch_size
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

        _, _, counts = get_stats(dataloader, mode, get_mean_std)
        
        label_counts_list_modes.append(counts)

    # check stratified 
    check_stratified(label_counts_list_modes)

    # should normalize weights
    if not cfg.training_setup.dataset.should_normalize_weights:
        counts = label_counts_list_modes[0]
        mean_label_count = np.mean(counts)  
        for count in counts:
            # if more or less than 10% of the mean label count
            if count < 0.9 * mean_label_count or count > 1.1 * mean_label_count:
                print("Warning: should probably normalize weights for training. Currently not set.")
                print("counts: ", counts, "mean_label_count: ", mean_label_count)
                break
    

def check_stratified(label_counts_list_modes):
    """
    Compares the percentage distribution of labels across different dataset modes.

    :param label_counts_list_modes: A list of label count arrays for each mode (train, val, test).
    """
    label_percentages = []

    # Calculate percentage distributions
    for counts in label_counts_list_modes:
        total_counts = sum(counts)
        percentages = [count / total_counts * 100 for count in counts]
        label_percentages.append(percentages)

    # Compare percentage distributions across modes
    reference_percentages = label_percentages[0]
    for percentages in label_percentages[1:]:
        assert np.allclose(percentages, reference_percentages, atol=3), "Label percentages differ more than 3% across modes."


def get_stats(
    dataloader: torch.utils.data.DataLoader,
    mode: str,
    augment: bool,
    get_mean_std: bool = False,
):
    list_images = []
    list_labels = []
    for i, out_dataloader in enumerate(dataloader):
        #print(i)
        images, labels = out_dataloader
        list_images.append(images.cpu().numpy())
        list_labels.append(labels.cpu().numpy())

    images = np.concatenate(list_images, axis=0)
    mean = np.mean(images, axis=(0, 2, 3)).tolist()
    std = np.std(images, axis=(0, 2, 3)).tolist()

    if get_mean_std:
        if mode == "train":
            print(f"mean: {mean:.8f}")
            print(f"std: {std:.8f}")
        else:
            # ignore the mean and std of the validation and test set
            pass
    else:
        # check if the mean and std are close to 0 and 1
        # can not check 

        if mode == "train" and not augment:
            for c in mean:
                assert np.isclose(c, 0.0, atol=0.1), f"mean should be zero mean: {mean}"
            for c in std:
                assert np.isclose(c, 1.0, atol=0.1), f"std should be one std: {std}"
        else:
            print("mode: ", mode)
            # print if the mean and std are not close to 0 and 1
            for c in mean:
                if not np.isclose(c, 0.0, atol=0.015):
                    print("mean: ", mean)
                    break
            for c in std:
                if not np.isclose(c, 1.0, atol=0.015):
                    print("std: ", std)
                    break

    
    labels = np.concatenate(list_labels, axis=0)
    values, counts = np.unique(labels, return_counts=True)

    return mean, std, counts

def hash_tensor(tensor):
    return hashlib.sha256(tensor.tobytes()).hexdigest()

if __name__ == "__main__":
    test_datamodule("imagenet", get_mean_std=False, no_augment=True)

