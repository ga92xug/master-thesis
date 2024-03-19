
import os
import pickle
import sys
from typing import Any, Dict, List, Union

import wandb

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.d_application.util import create_name_comparision_models
from src.optimization.optimization_utils import flatten_dict

"""
        name_network = flatten_run_config["network._target_"].split(".")[-1]
        # pretrained
        if "network.pre_trained" in flatten_run_config or "network.pretrained" in flatten_run_config:
            name_network += " pre-trained"
"""

def get_wandbdata_with_filters(
        wandb_entity: str,
        wandb_projects: str,
        filters: Dict[str, Any],
        #group_by: Union[str, List[str]] = None,
    ):
    api = wandb.Api()
    runs = api.runs(path=f"{wandb_entity}/{wandb_projects}", filters=filters)
    print("Runs:", len(runs))

    results = {}
    for run in runs:
        if run.state != "finished":
            print(f"Skipping run {run.id} because state is {run.state}")
            continue

        # group by
        group_by = create_name_comparision_models(run.config)

        if group_by not in results:
            results[group_by] = {}
        
        results[group_by][run.id] = download_run(run=run)
    return results

def download_run(
        run: wandb.sdk.wandb_run.Run = None,
    ) -> Dict[str, List[float]]:
    """
    Download the data for one run.
    """
    metrics = [
        "train/acc_weighted", "train/acc", 
        "valid/acc_weighted", "valid/acc", 
        "test/acc_weighted", "test/acc"
    ]

    result = {}
    
    for metric in metrics:
        if metric in run.history().columns:
            result[metric] = (run.history(keys=[metric]).values[:, 1] * 100).tolist()


    result["param_count"] = run.config["param_count"] * 1e6
    result["flops"] = run.config["GFLOPs_per_image"] * 1e9
        
    return result

def save_wandb_data(
        filters: Dict,
        wandb_projects: str,
        save_name: str,
        wandb_entity: str = "ga92xug",  
    ):
    results_for_filter = get_wandbdata_with_filters(
        wandb_entity=wandb_entity,
        wandb_projects=wandb_projects,
        filters=filters,
    )
    print("Results for filter:", results_for_filter)
    save_data(results_for_filter, save_name)
    return results_for_filter

def save_data(
        data: Dict,
        save_name: str,
        save_folder_name: str = "wandb_data",
    ):
    save_path = f"{save_folder_name}/{save_name}.pkl"
    os.makedirs(save_folder_name, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

def isic2019():
    filters = {
        "State": "finished",
        #"config.dataset.name": "isic2019",
    }
    wandb_projects = "isic2019"
    save_wandb_data(filters, wandb_projects, save_name=wandb_projects)

def camelyon17():
    filters = {
        "config.dataset.name": "camelyon17",
        "State": "finished",
    }
    wandb_projects = "domain_shift"
    save_wandb_data(filters, wandb_projects, save_name="camelyon17")


def derma():
    filters = {
        "State": "finished",
        "config.dataset.data.val_on": "derm7pt",
    }
    wandb_projects = "derma"
    save_wandb_data(filters, wandb_projects, save_name=wandb_projects)

if __name__ == "__main__":
    #isic2019()
    derma()
    #camelyon17()