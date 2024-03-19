import os
from typing import Dict, List, Optional, Tuple
import wandb
import sys

sys.path.append(f"{os.getcwd()}")
from src.optimization.optimization_utils import flatten_dict


def get_batch_size(run_config: dict):
    try:
        # this is the standard way
        batch_size = run_config["training"]["dataset"]["batch_size"]
    except KeyError:
        # legacy
        flatten_run_config = flatten_dict(run_config)
        batch_size = None
        for key, value in flatten_run_config.items():
            if "batch_size" in key:
                batch_size = value
                break

    assert batch_size is not None, "Could not find batch size in run config."
    
    return batch_size


def get_parameter_count(run: wandb.sdk.wandb_run.Run) -> float:
    """
    The setup in which we store the parameter count has changed over time.

    - Most recent in run.config
    - Older as param count in history
    """

    # newest way
    if "param_count" in run.config:
        result = run.config["param_count"] * 1e6
    else:
        result = run.history(keys=["param_count"]).values[0, 1] * 1e6

    return result


def get_flops_per_image(run: wandb.sdk.wandb_run.Run) -> float:
    """
    The setup in which we store the parameter count has changed over time.

    - Most recent in run.config
    - Older as param count in history
    """    
    # newest way
    if "GFLOPs_per_image" in run.config:
        result = run.config["GFLOPs_per_image"] * 1e9
    elif "GFLOPs" in run.history().columns:
        result = run.history(keys=['GFLOPs']).values[0, 1] * 1e9
        if result >= 50:
            run_config = run.config
            batch_size = get_batch_size(run_config)
            result = result / batch_size
    else:
        print(f"Run {run.id} does not have flops!")
        result = None
        result = lambda x: x / batch_size

    return result

def get_all_metrics(run: wandb.sdk.wandb_run.Run, metric_name: str):
    old_to_new = [
        "train/acc_weighted", "train/acc", 
        "valid/acc_weighted", "valid/acc", 
        "test/acc_weighted", "test/acc"
    ]
    converter = {}
    if "." in metric_name:
        # replace every / in by . but save as /
        for old in old_to_new:
            new = old.replace("/", ".")
            converter[new] = old
    else:
        converter = {old: old for old in old_to_new}
    
    result = {}

    for converted, old in converter.items():
        if old in run.history().columns:
            result[converted] = run.history(keys=[old]).values[:, 1] * 100
        else:
            result[converted] = None

    return result