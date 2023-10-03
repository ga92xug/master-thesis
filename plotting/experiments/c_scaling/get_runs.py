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
from plotting.experiments.c_scaling.util import baseline_and_scaling_exp_2_scaling_exp, transform_data_for_plot
from plotting.util import plot_init, download_run, save_plot
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

def aggregate_filter_data(data: dict, metric: str, window_size: int = 3):
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

def get_data(
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
    ):
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

    return data


def transform_data(data: dict, metric: str):
    # aggregate data data of different random seeds for 1 type
    data = aggregate_filter_data(data, metric)

    # transform data into format for plotting
    one_exp = baseline_and_scaling_exp_2_scaling_exp(data)
    flops, accuracy_values, labels = transform_data_for_plot(one_exp)

    return flops, accuracy_values, labels

def example_data():
    flops = [np.array(7.06396968e+10), np.array(1.44022375e+11), np.array(2.42780298e+11), np.array(3.66913477e+11)]
    accuracy_values = [np.array([68.1474785 , 66.86897278, 66.55477285]), np.array([68.89136434, 68.97262534, 67.66067346]), np.array([70.24774353, 69.6485877 , 69.03163393]), np.array([69.07265186, 69.29402153, 66.99118813])]
    labels = ['w1', 'w1.5', 'w2', 'w2.5']

    return flops, accuracy_values, labels


def main():
    cfg, save_folder_name = plot_init("c_scaling/individual/")
    wandb_entity = cfg.wandb.entity
    wandb_projects = "SL-Scaling"
    metric = "valid.acc_weighted"

    data = get_data(wandb_entity, wandb_projects, metric)
    flops, accuracy_values, labels = transform_data(data, metric)
    #flops, accuracy_values, labels = example_data()


    fig, ax = plt.subplots()
    create_subplot(ax, flops, accuracy_values, "FLOPs", "Weigthed Validation Accuracy", labels, True, color='b')

    save_plot(
        figure=fig,
        name="individual",
        folder_name=save_folder_name,
    )

if __name__ == "__main__":
    main()