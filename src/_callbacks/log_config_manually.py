from pathlib import Path
from typing import List, Tuple
from lightning import Callback
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import hydra

class Log_Config(Callback):
    """
    This is a wandb limitation config is not logged in offline runs :(.
    https://github.com/wandb/wandb/issues/6952
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def on_fit_start(self, trainer, pl_module):
        if not wandb.run or not wandb.run.offline:
            return
        
        artifact = wandb.Artifact(name="settings", type="config", metadata=OmegaConf.to_container(self.cfg, resolve=True))
        wandb.log_artifact(artifact, aliases=["latest-run"])
        return
