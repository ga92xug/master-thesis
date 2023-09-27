import os
from typing import Dict, List, Optional
import wandb


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
        entity: str, 
        projects: list, 
        run_id: str,
        metric: str = "valid.acc",
        name_param_count: str = "param_count",
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

    result = {}

    run = find_run(entity, projects, run_id)

    result[metric] = run.history(keys=[metric]).values[:, 1]
    result["param_count"] = run.history(keys=[name_param_count]).values[:, 1][0] * 1e6
        
    # some of the runs do not have flops
    try:
        result["flops"] = run.history(keys=['GFLOPs']).values[:, 1][0] * 1e9
    except:
        pass

    return result


def download_data(
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
            result[label][run_id] = download_run(entity, projects, run_id, metric, name_param_count)

    return result


def save_plot(figure, name, folder_name='figures/first_experiments'):
    """
    Safe a matplotlib figure to a file.
    """
    name = name.replace(' ', '_').lower()

    if not os.path.exists(f'{folder_name}'):
        os.makedirs(f'{folder_name}')

    figure.savefig(f'{folder_name}/{name}.png', dpi=300, bbox_inches = "tight")

