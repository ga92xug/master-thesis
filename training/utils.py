import os.path
import sqlite3
import pandas as pd
import numpy as np
import io
import datetime
import hydra
import signal

from typing import List, Tuple

import torch

#from models import *
from networks import *
from networks.util import get_param_count

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
        metrics = ", ".join([f"{key}: {value:.3f}" for key, value in metrics.items()])
        print(f'{metrics}, loss: {loss:.3f}')

def get_out_dataloader(out_dataloader: Tuple):
    if len(out_dataloader) == 2:
        x, t = out_dataloader
        meta_data = None
    elif len(out_dataloader) == 3:
        # domain shift
        x, t, meta_data = out_dataloader
    else:
        raise ValueError("Dataloader should return 2 or 3 values")
    
    return x, t, meta_data

########################################################################################################################
# Utilites to build paths and names in a standard way
########################################################################################################################


def out_path(cfg):
    path = cfg.other.output_path
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