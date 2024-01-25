import os
from typing import Callable, List, Optional, Tuple, Union
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
import wandb
from omegaconf import DictConfig
from wandb.sdk.lib import filenames

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class Distributed_Writer(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


