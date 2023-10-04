from math import comb
from typing import Dict, Any
from click import group
from omegaconf import OmegaConf
import wandb
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(f"{os.getcwd()}")

from plotting.experiments.c_scaling.plot_scaling import *
from plotting.experiments.c_scaling.util import baseline_and_scaling_exp_2_scaling_exp, split_dict2lists
from plotting.util import plot_init, download_run, save_plot
from plotting.plot_functions import get_metric_from_downloaded_data, transform_data_to_arrays
from networks.util import flatten_dict


def aggregate_filter_data(exp_data: dict, metric: str, window_size: int = 3):
    combined_data = {}

    for sub_exp_name, sub_exp_data in exp_data.items():
        print("experiment_class", sub_exp_name)
        performance_data = get_metric_from_downloaded_data(sub_exp_data, metric, "extract_last", window_size)
        flops_data = get_metric_from_downloaded_data(sub_exp_data, "flops", "equal", 1)

        combined_data[sub_exp_name] = {}
        for group_name, performance in performance_data.items():
            combined_data[sub_exp_name][group_name] = {
                metric: performance,
                "flops": flops_data[group_name],
            }

    return combined_data

def transform_data(
        exp_data: dict, 
        metric: str,
        label2sub_exp_name_dict: dict,
    ):
    # aggregate data of different random seeds for 1 type
    exp_data = aggregate_filter_data(exp_data, metric)

    # transform data into format for plotting
    one_exp = baseline_and_scaling_exp_2_scaling_exp(exp_data, label2sub_exp_name_dict)
    print("one_exp", one_exp)
    flops, accuracy_values, labels = split_dict2lists(one_exp, metric)

    return flops, accuracy_values, labels

def get_wandbdata_with_filters(
        wandb_entity: str,
        wandb_projects: str,
        filters: Dict[str, Any],
        metric: str,
        group_by: str or List[str] = None,
    ):
    api = wandb.Api()
    runs = api.runs(path=f"{wandb_entity}/{wandb_projects}", filters=filters)

    results = {}
    seeds = {}
    for run in runs:
        if run.state != "finished":
            print(f"Skipping run {run.id} because it is not finished.")
            continue

        flatten_run_config = flatten_dict(run.config, separator=".")

        if group_by is not None:
            if isinstance(group_by, list):

                group = []
                for key in group_by:
                    if key == "model.increase_blocks.2.num_new_blocks":
                        try:
                            # only shows the increase add 3 standard blocks
                            group.append(flatten_run_config[key] + 3)
                        except KeyError:
                            # standard is 3 blocks not every run has this key
                            group.append(3)
                    else:
                        group.append(flatten_run_config[key])

                group = tuple(group)
            else:
                group = flatten_run_config.get(group_by, None)
        else:
            group = "all"

        print("group", group)
        if group is None:
            print(f"Skipping run {run.id} because it has no group.")
            continue


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


def get_data_for_exp(
        sub_exp_dict: dict,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
    ):
    """
    Get all the data for 1 experiment. 1 experiment is a set of sub experiments with labels.
    Each sub experiment has a set of filters that are used to get the right data from wandb.
    Optionally, the data can be grouped by a certain key.
    A group would for example be the width coefficient of the model.
    """

    

    data = {}
    for sub_exp_name, value_dict in sub_exp_dict.items():
        filters = value_dict["filters"]
        group_by = value_dict["group_by"]

        print("filters", filters, type(filters))
        # add config.training.epochs: 120 to filters
        filters["config.training.epochs"] = 120
        
        results_for_filter = get_wandbdata_with_filters(
            wandb_entity=wandb_entity,
            wandb_projects=wandb_projects,
            filters=filters,
            metric=metric,
            group_by=group_by,
        )
        data[sub_exp_name] = results_for_filter

    return data


def one_plot(
        sub_exp_dict: dict,
        label2sub_exp_name_dict: dict,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
        save_folder_name: str,
        name: str,    
    ):
    """
    
    Args:
    - sub_exp_dict: a sub_exp is a sub experiment that is part of a larger experiment.\
        For example baseline_3blocks is a subexperiment of the width_scaling experiment.

    """

    exp_data = get_data_for_exp(sub_exp_dict, wandb_entity, wandb_projects, metric)
    flops, accuracy_values, labels = transform_data(exp_data, metric, label2sub_exp_name_dict)

    fig, ax = plt.subplots()
    create_subplot(ax, flops, accuracy_values, "FLOPs", "Weigthed Validation Accuracy", labels, True, color='b', connect_dots=False)

    save_plot(
        figure=fig,
        name=name,
        folder_name=save_folder_name,
    )

def main():
    cfg, save_folder_name = plot_init("c_scaling/individual/", override=True)
    wandb_entity = cfg.wandb.entity
    wandb_projects = "SL-Scaling"
    metric = "valid.acc_weighted"
    experiments_dict = OmegaConf.to_container(
        cfg.experiments.filter_groupby_dict, resolve=True, throw_on_missing=True)

    exp2labels = OmegaConf.to_container(
        cfg.experiments.plots, resolve=True, throw_on_missing=True)

    for exp_name, label2sub_exp_name_dict in exp2labels.items():
        # label2sub_exp_name_dict dict that holds labels for each sub experiment
        if exp_name != "depth_scaling":
            continue
        print(exp_name)

        # extract all sub experiments that belong to one experiment
        sub_exp_dict = {}
        for label, sub_exp_name in label2sub_exp_name_dict.items():
            sub_exp_dict[sub_exp_name] = experiments_dict[sub_exp_name]


        one_plot(
            sub_exp_dict=sub_exp_dict,
            label2sub_exp_name_dict=label2sub_exp_name_dict,
            wandb_entity=wandb_entity,
            wandb_projects=wandb_projects,
            metric=metric,
            save_folder_name=save_folder_name,  
            name=exp_name,  
        )





if __name__ == "__main__":
    main()