import os
from typing import Dict, List, Optional, Tuple
import wandb
from hydra import compose, initialize
import matplotlib.pyplot as plt

def plot_init(individual_location:str, override: bool = False):
    """
    Initializes all plot functions. Gets the config and sets the font size.
    """
    overrides = []
    if override:
        overrides = [f"experiments={individual_location.replace('/', '_')[:-1]} "]

    with initialize(config_path="conf", version_base="1.2"):
        cfg = compose(config_name="config", overrides=overrides)

    # Set the font size for labels, tick labels, and titles
    plt.rcParams.update({'font.size': cfg.fontsize})

    return cfg, cfg.save_folder + individual_location

def get_fig_size(fig_size : Tuple, textwidth_in: float = 5.78853, reduction: float = 1.0):
    """
    This function calculates the figure size in inches based on the textwidth of the latex document.
    """
    
    fig_width = textwidth_in * reduction
    
    # Calculate the ratio of the figure width to the figure height
    ratio = fig_size[0] / fig_size[1]

    # Calculate the figure height in inches
    fig_height = fig_width / ratio

    fig_size = (fig_width, fig_height)
    return fig_size

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
        run: wandb.apis.public.Run,
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
            result[label][run_id] = download_run(
                entity=entity, 
                projects=projects, 
                run_id=run_id,
                metric=metric,
                name_param_count=name_param_count,
            )

    return result


def save_plot(figure, name, folder_name='figures/first_experiments'):
    """
    Safe a matplotlib figure to a file.
    """
    name = name.replace(' ', '_').lower()

    if not os.path.exists(f'{folder_name}'):
        os.makedirs(f'{folder_name}')

    figure.savefig(f'{folder_name}/{name}.png', dpi=300, bbox_inches = "tight")

