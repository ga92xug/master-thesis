from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from lightning import LightningDataModule
from omegaconf import DictConfig, open_dict

import pytest
import torch

import rootutils

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

from src.data.mnist_datamodule import MNISTDataModule
from src.data.datamodule import DataModule
from tests.train.helpers.hydra_init import hydra_compose


'''
@pytest.mark.parametrize("batch_size", [32, 128])
def test_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size, dataset", [(32, "mnist"), (128, "cifar10")])
def test_datamodule(
    cfg_train: DictConfig, 
    batch_size: int,
    dataset: str,
) -> None:
    """Tests `DataModule` to verify that all the datasets can be loaded correctly.
      It checks that the necessary attributes were created 
      (e.g., the dataloader objects), and that dtypes and batch sizes 
      correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """

    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.training_setup.dataset.batch_size = batch_size
        #cfg_train.training_setup = dataset + "-training"

    dm: LightningDataModule = DataModule(cfg_train.training_setup.dataset)

    dm.prepare_data()

    assert not dm.train_set and not dm.val_set and not dm.test_set

    dm.setup()
    assert dm.train_set and dm.val_set and dm.test_set
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.train_set) + len(dm.val_set) + len(dm.test_set)

    for dataloader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        batch = next(iter(dataloader))
        x, y = batch
        assert len(x) == batch_size
        assert len(y) == batch_size
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
'''

@pytest.mark.parametrize("dataset", ["cifar10"])
def test_datamodule(
    dataset: str,
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
    cfg = hydra_compose(overrides)

    dm: LightningDataModule = DataModule(cfg.training_setup.dataset)

    dm.prepare_data()

    assert not dm.train_set and not dm.val_set and not dm.test_set

    dm.setup()
    assert dm.train_set and dm.val_set and dm.test_set
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.train_set) + len(dm.val_set) + len(dm.test_set)

    for dataloader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        batch = next(iter(dataloader))
        x, y = batch
        assert len(x) == batch_size
        assert len(y) == batch_size
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64


if __name__ == "__main__":
    test_datamodule("predefined/cifar10")