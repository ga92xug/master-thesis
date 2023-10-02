from math import comb
from typing import Dict, Any
from click import group
import wandb
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(f"{os.getcwd()}")

from plotting.experiments.c_scaling.plot_scaling import *
from plotting.util import plot_init, download_run
from plotting.plot_functions import get_metric_from_downloaded_data, transform_data_to_arrays
from networks.util import flatten_dict


def get_wandbdata_with_filters(
        wandb_entity: str,
        wandb_projects: str,
        filters: Dict[str, Any],
        metric: str,
        group_by: str = None,
    ):
    api = wandb.Api()
    runs = api.runs(path=f"{wandb_entity}/{wandb_projects}", filters=filters)

    results = {}
    seeds = {}
    for run in runs:
        flatten_run_config = flatten_dict(run.config, separator=".")

        group = flatten_run_config[group_by] if group_by is not None else "all"
        print("group", group)
        if group not in results:
            results[group] = {}
            seeds[group] = set()
        
        seed = flatten_run_config["other.seed"]
        if seed in seeds[group]:
            print(f"Skipping run {run.id} because it has the same seed as another run in the same group.")
            continue
        else:
            seeds[group].add(seed)

        results[group][run.id] = download_run(run=run, metric=metric)

    return results

def combine_data(data: dict, metric: str, window_size: int = 3):
    combined_data = {}

    for experiment_class, experiment_data in data.items():
        performance_data = get_metric_from_downloaded_data(experiment_data, metric, "extract_last", window_size)
        flops_data = get_metric_from_downloaded_data(experiment_data, "flops", "equal", 1)

        combined_data[experiment_class] = {}
        for label, performance in performance_data.items():
            combined_data[experiment_class][label] = {
                metric: performance,
                "flops": flops_data[label],
            }

    return combined_data


def main():
    cfg, save_folder_name = plot_init("c_scaling/individual/")
    wandb_entity = cfg.wandb.entity
    wandb_projects = "SL-Scaling"
    metric = "valid.acc_weighted"

    tags = {
        "baseline_isic2019": None, 
        "width_scaling": "model.width_coefficient"
    }
    
    data = {}
    for tag, group_by in tags.items():
        filters = {
            "config.training.epochs": 120,
            "tags": tag,
        }
        
        results_for_filter = get_wandbdata_with_filters(
            wandb_entity=wandb_entity,
            wandb_projects=wandb_projects,
            filters=filters,
            metric=metric,
            group_by=group_by,
        )
        data[tag] = results_for_filter

    combined_data = combine_data(data, metric)
    print(combined_data)


if __name__ == "__main__":
    main()