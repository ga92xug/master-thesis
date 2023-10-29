from copy import deepcopy
from math import comb
from typing import Dict, Any
from click import group
from matplotlib import colors
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory

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

    for path_name, path_data in exp_data.items():
        performance_data = get_metric_from_downloaded_data(path_data, metric, "extract_last", window_size)
        # we 1% in GFLOPs as absolute tolerance -> switch batch size made the FLOPs change
        kwargs = {"atol": 1e7}
        flops_data = get_metric_from_downloaded_data(path_data, "flops", "equal", 1, **kwargs)

        combined_data[path_name] = {}
        for group_name, performance in performance_data.items():
            combined_data[path_name][group_name] = {
                metric: performance,
                "flops": flops_data[group_name],
            }

    return combined_data

def transform_data(
        wandb_data: dict, 
        metric: str,
        exp_dict: dict,
    ):
    """
        exp_dict: dict
            paths: dict
                label_mask: path_name
            colors: list
            linestyles: list
    """
    # aggregate data of different random seeds for 1 type
    wandb_data = aggregate_filter_data(wandb_data, metric)

    # transform data into format for plotting
    one_exp = baseline_and_scaling_exp_2_scaling_exp(wandb_data, exp_dict)
    flops, accuracy_values, labels = split_dict2lists(one_exp, metric)
    legend_labels = list(one_exp.keys())

    return flops, accuracy_values, labels, legend_labels

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
            print(f"Skipping run {run.id} because state is {run.state}")
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

        if group is None:
            print(f"Skipping run {run.id} because it has no group.")
            continue


        if group not in results:
            results[group] = {}
            seeds[group] = set()
        
        seed = flatten_run_config["other.seed"]
        if seed in seeds[group]:
            pass
            #print(f"Skipping run {run.id} because it has the same seed as another run in the same group.")
            #continue
        else:
            seeds[group].add(seed)

        results[group][run.id] = download_run(run=run, metric=metric)

    return results

#
#@memory.cache
def get_data_for_exp(
        paths_dict: dict,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
    ):
    """
    Get all the data for 1 experiment. 1 experiment is a set of paths experiments with labels.
    Each path has a set of filters that are used to get the right data from wandb.
    Optionally, the data can be grouped by a certain key.
    A group would for example be the width coefficient of the model.
    """

    data = {}
    for path_name, value_dict in paths_dict.items():
        filters = value_dict["filters"]
        group_by = value_dict["group_by"]

        # add config.training.epochs: 120 to filters
        filters["config.training.epochs"] = 120
        
        results_for_filter = get_wandbdata_with_filters(
            wandb_entity=wandb_entity,
            wandb_projects=wandb_projects,
            filters=filters,
            metric=metric,
            group_by=group_by,
        )
        data[path_name] = results_for_filter

    return data


def scaling_plot_data(
        paths2filter_dict: dict,
        exp_dict: dict,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
        xlabel: str,
        ylabel: str,
        save_folder_name: str,
        name: str,    
        produce_plot: bool = True,
    ):
    """
    
    Args:
    - sub_exp_dict: a sub_exp is a sub experiment that is part of a larger experiment.\
        For example baseline_3blocks is a subexperiment of the width_scaling experiment.

    """

    wandb_data = get_data_for_exp(paths2filter_dict, wandb_entity, wandb_projects, metric)
    flops, accuracy_values, labels, legend_labels = transform_data(wandb_data, metric, exp_dict)

    colors = exp_dict["colors"]
    linestyles = exp_dict["linestyles"]

    kwargs = {
        "flops_lists": flops,
        "accuracy_values_lists": accuracy_values,
        "colors": colors,
        "linestyles": linestyles,
        "labels_lists": labels if name != "compound_scaling" else None,
        "legend_labels": legend_labels if name == "compound_scaling" else None,
    }

    
    if produce_plot:
        multipath_individual_scaling_plot(
            **kwargs,
            xlabel=xlabel,
            ylabel=ylabel,
            save_name = name,
            save_folder_name=save_folder_name,
        )
    return kwargs


def produce_all_scaling_plots(
        cfg: DictConfig,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
        save_folder_name: str,
        xlabel: str = "GFLOPs",
        ylabel: str = "ISIC 2019 Valid Acc (%)",
        combined_plot: bool = False, 
    ):
    """
    exp_dict: dict
        paths: dict
            label_mask: path_name
        colors: list
        linestyles: list

    filter_dict: dict
        path_name: dict
            filters: dict
            group_by: str or list
    """

    filter_dict = OmegaConf.to_container(
        cfg.experiments.filter_groupby_dict, resolve=True, throw_on_missing=True)

    exp2labels = OmegaConf.to_container(
        cfg.experiments.plots, resolve=True, throw_on_missing=True)

    combined_plot_dict = {}

    for exp_name, exp_dict in exp2labels.items():
        #if exp_name != "compound_scaling":
        #    continue
        print(exp_name)

        # extract all sub experiments that belong to one experiment
        paths2filter_dict = {}
        for label_mask, path_name in exp_dict["paths"].items():
            paths2filter_dict[path_name] = filter_dict[path_name]


        save_dict = scaling_plot_data(
            paths2filter_dict=paths2filter_dict,
            exp_dict=exp_dict,
            wandb_entity=wandb_entity,
            wandb_projects=wandb_projects,
            metric=metric,
            save_folder_name=save_folder_name,  
            name=exp_name,  
            xlabel=xlabel,
            ylabel=ylabel,
        )

        if combined_plot and exp_name != "compound_scaling":
            combined_plot_dict[exp_name] = save_dict

    
    if combined_plot:
        print("combined_plot")
        combined_individual_scaling(
            combined_plot_dict,
            xlabel=xlabel,
            ylabel=ylabel,
            save_folder_name=save_folder_name,
            sharey=True,
        )


def main():
    cache_dir = './home/frischs/cache/'
    memory = Memory(location=cache_dir, verbose=1)
    cfg, save_folder_name = plot_init("c_scaling/individual/", override=True)
    wandb_entity = cfg.wandb.entity
    wandb_projects = "SL-Scaling"
    metric = "valid.acc_weighted"

    produce_all_scaling_plots(
        cfg=cfg,
        wandb_entity=wandb_entity,
        wandb_projects=wandb_projects,
        metric=metric,
        save_folder_name=save_folder_name, 
        combined_plot=True,
    )




if __name__ == "__main__":
    main()