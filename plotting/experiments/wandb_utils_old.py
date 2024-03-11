import os
from typing import Dict, List, Optional, Tuple
from joblib import Memory
import wandb
from hydra import compose, initialize
import matplotlib.pyplot as plt
import sys

sys.path.append(f"{os.getcwd()}")
from src.networks.util import flatten_dict


def find_run(entity: str, projects: list, run_id: str) -> wandb.apis.public.Run:
    """
    Find a run with the given `run_id` in a list of `projects`.

    Args:
        entity (str): The entity associated with the projects.
        projects (list): A list of project names to search in.
        run_id (str): The ID of the run to find.

    Returns:
        wandb.apis.public.Run: The found run.

    Raises:
        ValueError: If the run is not found in any of the projects.
    """
    if isinstance(projects, str):
        projects = [projects]

    for project in projects:
        try:
            run = wandb.Api().run(f"{entity}/{project}/{run_id}")
            return run
        except wandb.Error:
            continue

    raise ValueError(f"Run {run_id} not found in any of the projects.")

def download_run(
        run: wandb.sdk.wandb_run.Run = None,
        entity: str = None, 
        projects: list = None, 
        run_id: str = None,
        metric: str = "valid.acc",
        name_param_count: str = "param_count",
        normalize_flops: bool = True,
    ) -> Dict[str, List[float]]:
    """
    Download the data for one run.
    
    Parameters:
    - entity (str): The entity associated with the projects.
    - projects (list): A list of project names to search in.
    - run_id (str): The ID of the run to find.
    - metric (str): The metric to download (valid.acc, valid.acc_weighted).
    - name_param_count (str): (param_count, total_params).
    """

    assert run is not None or (entity is not None and projects is not None and run_id is not None), \
        "Either run or entity, projects, and run_id must be given."

    if run is None:
        run = find_run(entity, projects, run_id)

    if run.state != "finished":
        raise ValueError(f"Run {run_id} is not finished.")

    result = {}

    result[metric] = run.history(keys=[metric]).values[:, 1] * 100

    result["param_count"] = run.history(keys=[name_param_count]).values[:, 1][0] 
    if name_param_count == "param_count":
        result["param_count"] *= 1e6
        
    # some of the runs do not have flops
    if "GFLOPs" in run.history().columns:
        result["flops"] = run.history(keys=['GFLOPs']).values[:, 1][0] * 1e9
    else:
        print(f"Run {run_id} does not have flops!")
        result["flops"] = None

    if normalize_flops:
        run_config = run.config
        batch_size = get_batch_size(run_config)
        if result["flops"] is not None:
            result["flops"] = result["flops"] / batch_size
        else:
            result["flops"] = lambda x: x / batch_size
        
    return result


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



#cache_dir = '/home/frischs/.cache/get_wandb_data_multiple_runs/'
#memory = Memory(location=cache_dir, verbose=1)
#memory.clear(warn=False)  # Set warn=False to suppress warning messages
#@memory.cache
def get_wandb_data_multiple_runs(
        entity: str, 
        projects: str, 
        labels_run_ids: Dict[str, List[str]],
        name_param_count : str = "param_count", 
        metric : str = "valid.acc",
    ) -> Dict[str, Dict[str, Dict]]: 
    """
    Download the data for multiple runs. \
    Each run is associated with a label one label can have 1 or more runs in the list. \
    
    Parameters:
    - entity (str): The entity associated with the projects.
    - projects (list): A list of project names to search in.
    - labels_run_ids (Dict[str, List[str]]): A dictionary of labels each label has a list of 1 or more run ids.
    - name_param_count (str): (param_count, total_params).
    - metric (str): The metric to download (valid.acc, valid.acc_weighted).
    """

    result = {}
    for label, run_ids in labels_run_ids.items():
        assert isinstance(run_ids, list), "run_ids must be a list"
        result[label] = {}
        
        # download the data for each run id associated with one label
        for run_id in run_ids:
            result[label][run_id] = download_run(
                entity=entity, 
                projects=projects, 
                run_id=run_id,
                metric=metric,
                name_param_count=name_param_count,
            )

    return result

