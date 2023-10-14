import os.path
import sqlite3
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import io
import datetime
import hydra
import signal
import os
import torch

from typing import List, Tuple, Dict, Any, Union

import torch
import wandb

#from models import *
from networks import *
from networks.util import flatten_dict, get_param_count

################################################################################
# building the model
################################################################################



def build_model(cfg, n_inputs, n_outputs, device, log, is_nas=False, trial_data=None):
    if not is_nas:
        # normal model building

        # build the model
        model = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            image_size=cfg.dataset.resolution,
        ).to(device)
        if device != torch.device("cpu"):
            model = torch.nn.DataParallel(model)
        if cfg.training.compile:
            model = torch.compile(model)
        print("Stage 2: model built")

    else:
        # NAS model building
        try:
            # Your command that builds the model
            # Place the command here that occasionally takes a long time

            # build the model
            model = hydra.utils.instantiate(
                cfg.model,
                input_channels=n_inputs,
                num_classes=n_outputs,
                image_size=cfg.dataset.resolution,
            ).to(device)
            if device != torch.device("cpu"):
                model = torch.nn.DataParallel(model)
            if cfg.training.compile:
                model = torch.compile(model)

            # Cancel the alarm since the command finished before the timeout
            signal.alarm(0)
        except TimeoutError:
            # Handle the timeout error
            
            print("Command execution timed out")

        except torch.cuda.CudaError:
            print("Cuda out of memory")



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


def print_results(metrics, loss, duration, mode, epoch, verbose):
    if verbose:
        print('-'*100)
        print(f'{mode} Epoch: {epoch} lasted {duration:.3f} seconds')
        metrics = ", ".join([f"{key}: {value:.3f}" for key, value in metrics.items() if key not in ["duration", "loss"]])
        print(f'{metrics}, loss: {loss:.3f}')

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
# Utilites to build paths and names in a standard way
########################################################################################################################

def give_wandb_name(
        model: torch.nn.Module, 
        wandb_run: wandb.sdk.wandb_run.Run,
        ignore_name: bool = False,
    ) -> None:
    """
    Updates the wandb_run name.
    """
    config = wandb_run.config
    give_name = config["wandb"]["give_name"]
    project = config["wandb"]["project"]
    model_name = config["model"]["_target_"]

    if not give_name:
        return

    if isinstance(give_name, str) and not ignore_name:
        wandb_run.name = give_name
    elif project == "SL-Scaling" and "EquivariantNASNet" in model_name:
        model_config = config["model"]
        # special naming convention for Scaling
        num_blocks = model_config.get("num_blocks", None)
        # increase_blocks.2
        if num_blocks is None:
            try:
                num_blocks = model_config["increase_blocks"]["2"]["num_new_blocks"] + 3
            except KeyError:
                num_blocks = 3

        depth_coefficient = model_config["depth_coefficient"]
        width_coefficient = model_config["width_coefficient"]
        resolution = config["training"]["dataset"]["resolution"]
        wandb_run.name = f"b{num_blocks}_d{depth_coefficient}_w{width_coefficient}_r{resolution}"
    else:
        wandb_run.name = model.name


def output_path(path: str):
    """
    Returns the path to the output folder.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    return path
    



def plot_path(config):
    return os.path.join(out_path(config), exp_name(config) + ".svg")


def backup_path(config):
    backup_folder = os.path.join(out_path(config), exp_name(config))
    return os.path.join(backup_folder, f"_{config.other.seed}.model")

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
    def __init__(self, monitor: str, mode: str, patience: int, min_delta: float, save_path: str, verbose: int = 1, store_in_memory: bool=True):
        """
        Initialize the EarlyStopping class.

        Args:
        - monitor (str): The metric name to monitor.
        - mode (str): One of {'min', 'max'}. Whether to minimize or maximize the monitor metric.
        - patience (int): Number of epochs with no improvement to wait before early stopping.
        - min_delta (float): Minimum change in the monitor metric to qualify as improvement.
        - save_path (str): Directory to save the best model.
        - store_in_memory (bool): Whether to store the best model in memory or on disk.
        """
        assert mode in ['min', 'max'], "Mode must be one of {'min', 'max'}."
        assert monitor.split(".")[0] == "valid" and monitor.split(".")[1] in ["loss", "acc", "acc_weighted"], "Monitor must be one of {'valid.loss', 'valid.acc', 'valid.acc_weighted'}."

        self.monitor = monitor.split(".")[1]
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.verbose = verbose
        self.best_score = None
        self.save_path = os.path.join(save_path, "best_model.pth")
        self.store_in_memory = store_in_memory
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def should_stop(self, metrics: Dict[str, Any], model: torch.nn.Module) -> bool:
        """
        Check if early stopping should be executed.

        Args:
        - metrics (dict): A dictionary of current epoch metrics.
        - model (torch.nn.Module): The current model.

        Returns:
        - bool: True if early stopping should be executed, False otherwise.
        """
        current_score = metrics.get(self.monitor, None).item()
        
        if current_score is None:
            raise ValueError(f"Monitor {self.monitor} does not exist in metrics.")

        if self.best_score is None:
            self.best_score = current_score
            self.save_model(model)
            return False

        if ((self.mode == 'min' and current_score < (self.best_score - self.min_delta)) or
            (self.mode == 'max' and current_score > (self.best_score + self.min_delta))):
            self.best_score = current_score
            self.counter = 0
            self.save_model(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.verbose:
                print(f"Early stopping. No improvement in {self.monitor} for {self.patience} epochs.")
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

