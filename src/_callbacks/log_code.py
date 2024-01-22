import os
from typing import Callable, List, Optional, Tuple, Union
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
import wandb
from omegaconf import DictConfig
from wandb.sdk.lib import filenames


class Log_Code(Callback):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        if trainer.logger is None:
            return
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                wandb.run.log_code(
                    root=".", 
                    include_fn=lambda path: path.endswith(".py") or 
                        path.endswith(".yaml"),
                    exclude_fn=lambda path: False,
                )
                break

        return