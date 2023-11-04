import os
from typing import Dict, List, Optional, Tuple
import wandb
from hydra import compose, initialize
import matplotlib.pyplot as plt

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
    result["param_count"] = run.history(keys=[name_param_count]).values[:, 1][0] * 1e6
        
    # some of the runs do not have flops
    try:
        result["flops"] = run.history(keys=['GFLOPs']).values[:, 1][0] * 1e9

        # normalize flops
        if normalize_flops:
            run_config = run.config
            batch_size = run_config["training"]["dataset"]["batch_size"]
            result["flops"] = result["flops"] / batch_size
    except:
        pass

    return result


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

