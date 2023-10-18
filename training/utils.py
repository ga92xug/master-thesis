import os.path
import sqlite3
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import io
import datetime
import hydra
import signal
import os
import torch

from typing import List, Optional, Tuple, Dict, Any, Union

import torch
import wandb
from ray.air.integrations.wandb import setup_wandb

from networks import *
from networks.eq_nasnet.naming_eq_nasnet import get_scaling_name


################################################################################
# building the model
################################################################################

def debug_run(cfg: DictConfig):
    if cfg.other.debug:
        print("Debug mode")
        cfg.training.epochs = 1
        cfg.training.steps_per_epoch = 10
        cfg.wandb.mode = "disabled"


def allowed_usage_time(
        respect_start_time: bool,
        start_time: datetime.time = datetime.time(hour=8, minute=30),
        end_time: datetime.time = datetime.time(hour=20),
):
    """
    GPU sharing. Check if the current time is within the allowed usage time.
    """
    if not respect_start_time:
        return

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).time()

def get_out_dataloader(
        out_dataloader: Tuple, 
        device: str = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(out_dataloader) == 2:
        x, t = out_dataloader
        meta_data = None
    elif len(out_dataloader) == 3:
        # domain shift
        x, t, meta_data = out_dataloader
    else:
        raise ValueError("Dataloader should return 2 or 3 values")
    
    x = x.to(device)
    t = t.to(device)

    return x, t, meta_data

########################################################################################################################
# Paths and Names
########################################################################################################################

def init_wandb(cfg: DictConfig):
    if cfg.NAS.trial_index == -1:
        # normal training mode
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        kwargs_wandb = {
            "config": wandb_config,
            "project": cfg.wandb.project,
            "mode": cfg.wandb.mode,
            "notes": cfg.wandb.notes,
            "tags": cfg.wandb.tags,
        }
        
        if cfg.ray:
            print("init ray wandb", kwargs_wandb)
            wandb_run = setup_wandb(rank_zero_only=False, **kwargs_wandb)
            print("Wandb run initialized:", wandb_run, type(wandb_run))
        else:
            wandb_run = wandb.init(project=cfg.wandb.project, config=wandb_config, \
            mode=cfg.wandb.mode, notes=cfg.wandb.notes, tags=cfg.wandb.tags)

            # merge wandb config with cfg. Sweep bug https://github.com/wandb/wandb/issues/4686
            cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(wandb.config)))
            wandb_update_config(cfg, wandb_run)

        wandb_run.log_code(".")
    else:
        wandb_run = None

    return wandb_run

def wandb_update_config(cfg: DictConfig, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Recursively updates the wandb config with the nested DictConfig object.
    We can not use wandb.config.update(cfg) because during sweeps some keys are not allowed to be updated.

    Args:
    - cfg (DictConfig): The nested DictConfig object.
    - wandb_run (wandb.sdk.wandb_run.Run): The wandb run object.
    """
    def recursive_update(cfg_dict, wandb_config_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                if key not in wandb_config_dict:
                    wandb_config_dict[key] = {}
                recursive_update(value, wandb_config_dict[key])
            else:
                wandb_config_dict[key] = value

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    recursive_update(cfg_dict, wandb_run.config)

def give_wandb_name(
        model: torch.nn.Module, 
        wandb_run: wandb.sdk.wandb_run.Run,
        ignore_name: bool = False,
    ) -> None:
    """
    Updates the wandb_run name.
    """
    if wandb_run is None:
        return
    
    config = wandb_run.config
    give_name = config["wandb"]["give_name"]
    project = config["wandb"]["project"]
    model_name = config["model"]["_target_"]

    if not give_name:
        return

    if isinstance(give_name, str) and not ignore_name:
        wandb_run.name = give_name
    elif project == "SL-Scaling" and "EquivariantNASNet" in model_name:
        wandb_run.name = get_scaling_name(config)
    else:
        wandb_run.name = model.name




def output_path(path: str):
    """
    Returns the path to the output folder.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    return path
    
def backup_path(should_backup, output_path: str, save_id: str, verbose: int) -> str:
    if not should_backup:
        return None

    model_path = os.path.join(output_path, save_id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if verbose > 2:
        print(f"Model path: {model_path}")

    model_path = os.path.join(model_path, "model.pth")
    return model_path
    

def allowed_usage_time(
    gpu_time_limit: bool,
    start_time: datetime.time = datetime.time(hour=8, minute=30),
    end_time: datetime.time = datetime.time(hour=20),
):
    """
    GPU sharing. Check if the current time is within the allowed usage time.
    """
    if not gpu_time_limit:
        return
    # Get the current time in GMT+2
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).time()

    # Check if the current time is within the range
    if start_time <= now <= end_time:
        raise ValueError("GPU usage not allowed between 8am and 8pm GMT+2")



######################################################
# Early stopping
######################################################

class EarlyStopping:
    def __init__(
            self, 
            monitor: str, 
            mode: str, 
            patience: int, 
            min_delta: float, 
            save_path: str, 
            verbose: int = 1,
            baseline: Optional[float] = None, 
            store_in_memory: bool=True
        ):
        """
        Initialize the EarlyStopping class.

        Args:
        - monitor (str): The metric name to monitor.
        - mode (str): One of {'min', 'max'}. Whether to minimize or maximize the monitor metric.
        - patience (int): Number of epochs with no improvement to wait before early stopping.
        - min_delta (float): Minimum change in the monitor metric to qualify as improvement.
        - save_path (str): Directory to save the best model.
        - verbose (int): Verbosity level.
        - baseline (float): Baseline value for the monitor metric. Early stopping will only start after the metric surpasses the baseline.
        - store_in_memory (bool): Whether to store the best model in memory or on disk.
        """
        assert mode in ['min', 'max'], "Mode must be one of {'min', 'max'}."
        assert monitor.split(".")[0] == "valid" and monitor.split(".")[1] in ["loss", "acc", "acc_weighted"], "Monitor must be one of {'valid.loss', 'valid.acc', 'valid.acc_weighted'}."

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.verbose = verbose
        self.baseline = baseline
        self.best_score = None
        self.best_epoch = -1
        self.save_path = save_path
        self.store_in_memory = store_in_memory
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def should_stop(
            self, 
            metrics: Dict[str, Any], 
            model: torch.nn.Module,
            epoch: int
        ) -> bool:
        """
        Check if early stopping should be executed.

        Args:
        - metrics (dict): A dictionary of current epoch metrics.
        - model (torch.nn.Module): The current model.

        Returns:
        - bool: True if early stopping should be executed, False otherwise.
        """
        current_score = metrics.get(self.monitor.split(".")[1], None).item()
        
        if current_score is None:
            raise ValueError(f"Monitor {self.monitor} does not exist in metrics.")

        # Don't start counting patience until the metric surpasses the baseline
        if self.baseline is not None:
            if (self.mode == 'min' and current_score > self.baseline) or \
               (self.mode == 'max' and current_score < self.baseline):
                return False

        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.save_model(model)
            return False

        if ((self.mode == 'min' and current_score < (self.best_score - self.min_delta)) or
            (self.mode == 'max' and current_score > (self.best_score + self.min_delta))):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self.save_model(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            try:
                log_dict = {
                    "early_stop": {
                        "best_epoch": self.best_epoch, 
                        f"{self.monitor}_best": self.best_score
                    }
                }
                wandb.log(log_dict)
            except:
                # wandb not initialized
                pass

            if self.verbose:
                print(f"Early stopping in {epoch}. No improvement in {self.monitor} for {self.patience} epochs.")
            return True
        return False

    def save_model(self, model: torch.nn.Module) -> None:
        """
        Save the model. If store_in_memory is True, the model is stored in memory. Otherwise, it is stored on disk.

        Args:
        - model (torch.nn.Module): The model to save.
        """
        if self.store_in_memory:
            self.weights = model.state_dict()
        else:
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            torch.save(model.state_dict(), self.save_path)

    def restore_best_weights(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Restore the best weights for the given model.

        Args:
        - model (torch.nn.Module): The model whose weights will be restored.

        Returns:
        - torch.nn.Module: The model with the best weights restored.
        """
        if self.store_in_memory:
            model.load_state_dict(self.weights)
        else:
            model.load_state_dict(torch.load(self.save_path))
        return model

