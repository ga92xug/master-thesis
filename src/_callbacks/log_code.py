from typing import List, Tuple
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
import wandb
from omegaconf import DictConfig


class Log_Code(Callback):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        #for logger in trainer.loggers:
        #    if isinstance(logger, WandbLogger):
        #        pass
        wandb.run.log_code(".")

        return
