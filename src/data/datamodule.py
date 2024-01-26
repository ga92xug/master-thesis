from typing import Any, Dict, Optional, Tuple

import hydra
from omegaconf import DictConfig
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from src.data._transforms.cut_mix import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from src.data.ra_sampler import RASampler

class DataModule(LightningDataModule):
    """`LightningDataModule`

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        is_dist: bool = False,
    ) -> None:
        """Initialize a `DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        #print(self.hparams)

        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

        self.batch_size_per_device = data_cfg.batch_size
        self.weights = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.
        """
        return self.hparams.data_cfg.num_classes

    @property
    def num_channels(self) -> int:
        """Get the number of classes.
        """
        return self.hparams.data_cfg.num_channels

    @property
    def image_size(self) -> int:
        """Get the image size in int (assumes square images).
        """
        return self.hparams.data_cfg.resolution
    
    @property
    def normalization_weights(self) -> torch.Tensor:
        """Returns the normalization weights for the loss function. \
            If they are not yet computed a RuntimeError is raised.

        :return: normalization weights 
        """
        if not self.hparams.data_cfg.should_normalize_weights \
            or self.hparams.data_cfg.reduction_factor > 1:
            # if we overfit on few training samples, we do not need to normalize the weights
            return None

        if self.weights is None:
            self.setup()

        assert isinstance(self.weights, torch.Tensor) or self.weights is None, "weights should be a tensor or None"
        return self.weights

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # We download the data manually since not all datasets are supported by torchvision.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.data_cfg.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.data_cfg.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.data_cfg.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.train_set and not self.val_set and not self.test_set:
            datasets, self.weights, self.dataloader_kwargs = hydra.utils.instantiate(
                self.hparams.data_cfg,
            )

            self.train_set = datasets["train"]
            self.val_set = datasets["valid"]
            self.test_set = datasets["test"]

            if self.hparams.data_cfg.test_as_valid:
                # swap val_loader and test_loader to check generalization early
                self.val_set, self.test_set = self.test_set, self.val_set

            self.setup_mixup_cutmix()
        else:
            print("Datasets already loaded!")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.hparams.is_dist:
            if self.hparams.data_cfg.get("ra_sampler", False):
                train_sampler = RASampler(self.train_set, shuffle=True, repetitions=self.hparams.data_cfg.ra_reps)
            else:
                train_sampler = DistributedSampler(self.train_set)
        else:
            train_sampler = RandomSampler(self.train_set)

        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.data_cfg.workers,
            pin_memory=self.hparams.data_cfg.pin_memory,
            persistent_workers=self.hparams.data_cfg.persistent_workers,
            sampler=train_sampler,
            **self.dataloader_kwargs.get("train", {}),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """

        if self.hparams.is_dist:
            val_sampler = DistributedSampler(self.val_set, shuffle=False)
        else:
            val_sampler = SequentialSampler(self.val_set)

        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.data_cfg.workers,
            pin_memory=self.hparams.data_cfg.pin_memory,
            persistent_workers=self.hparams.data_cfg.persistent_workers,
            sampler=val_sampler,
            **self.dataloader_kwargs.get("valid", {}),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.data_cfg.workers,
            pin_memory=self.hparams.data_cfg.pin_memory,
            shuffle=False,
            **self.dataloader_kwargs.get("test", {}),
        )
    
    def predict_dataloader(self) -> DataLoader[Any]:
        """ Return the test dataloader. """
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


    def setup_mixup_cutmix(self):
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=self.hparams.get("mixup_alpha", 0), 
            cutmix_alpha=self.hparams.get("cutmix_alpha", 0),
            num_categories=self.num_classes, 
            #use_v2=args.use_v2
        )
        if mixup_cutmix is not None:
            train_data_loader_kwargs = self.dataloader_kwargs.get("train", {})
            if "collate_fn" in train_data_loader_kwargs:
                _default_collate = train_data_loader_kwargs["collate_fn"]
            else:
                _default_collacte = default_collate
                
            def collate_fn(batch):
                return mixup_cutmix(*_default_collate(batch))
            
            train_data_loader_kwargs["collate_fn"] = collate_fn
            self.dataloader_kwargs["train"] = train_data_loader_kwargs

