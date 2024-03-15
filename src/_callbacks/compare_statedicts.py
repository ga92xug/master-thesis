import os
from typing import Any, Callable, List, Optional, Tuple, Union
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import wandb
from omegaconf import DictConfig
from wandb.sdk.lib import filenames

from src.utils.utils import compare_state_dicts


class CompareModelCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if batch_idx < 1:
            return

        # Save the model's state dict
        torch.save(pl_module.state_dict(), 'temp_model.pth')

        # Create a new instance of the model
        new_model = pl_module.__class__(**pl_module.hparams)  # Replace with appropriate arguments for your model
        new_model.load_state_dict(torch.load('temp_model.pth'))

        # Compare the state dicts
        compare_state_dicts(pl_module, new_model)

        # Optionally, delete the temporary file
        os.remove('temp_model.pth')

        quit()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Save the model's state dict
        torch.save(pl_module.state_dict(), 'temp_model.pth')

        # Create a new instance of the model
        new_model = pl_module.__class__(**pl_module.hparams)
        new_model.load_state_dict(torch.load('temp_model.pth'))

        # Compare the state dicts
        compare_state_dicts(pl_module, new_model)

        # Optionally, delete the temporary file
        os.remove('temp_model.pth')

        # run eval and test loop
        #trainer.validate(model=model, datamodule=trainer.)