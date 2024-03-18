from typing import List, Dict
import numpy as np
import re

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
from plotting.experiments.wandb_utils import download_run
from plotting.experiments.plot_acc_flops_params import get_metric_from_downloaded_data, transform_data_to_arrays
from networks.util import flatten_dict


cache_dir = '/home/frischs/.cache/scaling_plots/'
memory = Memory(location=cache_dir, verbose=1)
@memory.cache
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


def aggregate_filter_data(exp_data: dict, metric: str, window_size: int = 3):
    """
    Aggregate the data for each path.
    """

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

def changes_for_paper(
        flops: list,
        accuracy_values: list,
        labels: list,
        legends: list,
        name: str,
    ):
    """
    Changes to the data for the paper.
    - Depth Scaling: 
        - die Werte raus, die die Accuracy droppen
        - legend: statt b3 und b4 bei den Punkten einfach eine Legende die die zwei Farben beschreibt
    - Compound Scaling:
        - ohne b4_d-1-3-3-3
    """
    def legend_gen(labels):
        legends = []
        # change the legend
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if j == 0:
                    if "b3" in labels[i][j]:
                        legends.append("3 Blocks")
                    elif "b4" in labels[i][j]:
                        legends.append("4 Blocks")
                    else:
                        raise ValueError("Unknown label")

                # remove the b3 and b4 from the labels
                labels[i][j] = labels[i][j].replace("b3", "")
                labels[i][j] = labels[i][j].replace("b4", "") 
        
        return labels, legends


    if name == "depth_scaling":
        # remove the data points that drop the accuracy

        for j in range(len(labels)):
            # iterate over all paths
            i = 0
            while i < len(labels[j]):
                remove_labels = ["d=1-3-3-3", "d=1-4-4-4", "d=1-3-3", "d=1-4-4"]

                for remove_label in remove_labels:
                    if remove_label in labels[j][i]:
                        flops[j].pop(i)
                        accuracy_values[j].pop(i)
                        labels[j].pop(i)
                        i -= 1

                i += 1            

        labels, legends = legend_gen(labels)
    elif name == "resolution_scaling":
        labels, legends = legend_gen(labels)
    elif name == "compound_scaling":
        # done in config
        for i, _ in enumerate(legends):
            legends[i] = legends[i].replace("_", " ")
            legends[i] = legends[i].replace("b3", "3 Blocks")
            legends[i] = legends[i].replace("b4", "4 Blocks")
            legends[i] = legends[i].replace("d-", "d=")
            legends[i] = legends[i].replace("r", "r=")
            legends[i] = legends[i].replace("w", "w=")    
    else:
        legends = None

    return flops, accuracy_values, labels, legends
                  

def transform_data(
        wandb_data: dict, 
        metric: str,
        exp_dict: dict,
        name: str,
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
    one_exp = process_raw_wandb_data(wandb_data, exp_dict)
    flops, accuracy_values, labels = split_dict2lists(one_exp, metric)

    legend = list(one_exp.keys())

    flops, accuracy_values, labels, legend = changes_for_paper(
        flops=flops, 
        accuracy_values=accuracy_values, 
        labels=labels,
        legends=legend, 
        name=name
    )

    return flops, accuracy_values, labels, legend

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


def replace_first_occurrence(input_str, search_str, replace_str):
    index = input_str.find(search_str)
    if index != -1:
        return input_str[:index] + replace_str + input_str[index + len(search_str):]
    return input_str

def extract_number(key):
    match = re.search(r'\[(.*)\]', key)
    if match:
        number = re.sub(r'[^0-9.]+', '', match.group(1))
        return float(number)
    else:
        return None
    
            
def process_raw_wandb_data(
        wandb_data: dict,
        exp_dict: Dict[str, str],
    ):
    
    """
    Transforms the input data into the format required for create_subplot.

    Args:
        exp_dict: dict
            paths: dict
                label_mask: path_name
            colors: list
            linestyles: list


    Returns:
    transformed_data: A dictionary with transformed data.
    """
    transformed_data = {}
    # inverse the dictionary
    path2lable_mask = {v: k for k, v in exp_dict["paths"].items()}

    #print("exp_data", exp_data)

    for path_name, path_wandb_data in wandb_data.items():
        label_mask = path2lable_mask[path_name]

        transformed_data[path_name] = {}
        for group_name, group_dict in path_wandb_data.items():
            if len(path_wandb_data) == 1:
                # these paths do not have different variations like w1.5,w2,w2.5
                label = label_mask
            else:
                if isinstance(group_name, tuple):
                    label = label_mask
                    for group in group_name:
                        label = label_mask.replace("X", str(group))
                        #label = replace_first_occurrence(label, "X", str(group))
                elif isinstance(group_name, (str, int, float)):
                    label = label_mask.replace("X", "[" + str(group_name) + "]")
                else:
                    raise ValueError(f"Unknown type of group_name: {type(group_name)}")

            if label in transformed_data[path_name]:
                raise ValueError(f"Label {label} already exists in transformed_data.")
            transformed_data[path_name][label] = group_dict

    sorted_dict = {}
    for k, v in transformed_data.items():
        sorted_dict[k] = dict(sorted(v.items(), key=lambda item: extract_number(item[0])))

    sorted_dict = rename_dict_keys(sorted_dict)
    return sorted_dict

def rename_dict_keys(old_dict: dict):
    """
    Renames the keys of the dictionary. Not inplace.
    """
    new_dict = {}
    for k, v in old_dict.items():
        new_v = {}
        for label, values in v.items():
            new_label = label.replace("[", "=").replace("]", "")
            new_v[new_label] = values
        new_dict[k] = new_v
    return new_dict

def split_dict2lists(data, metric: str):
    """
    Splits the dictionary into lists
    """

    flop_list = []
    accuracy_list = []
    label_list = []

    for path, path_dict in data.items():
        flop = []
        accuracy = []
        labels = []
        for label, values in path_dict.items():
            flop.append(values['flops'])
            accuracy.append(values[metric])
            labels.append(label)

        flop_list.append(flop)
        accuracy_list.append(accuracy)
        label_list.append(labels)

    return flop_list, accuracy_list, label_list


def example_data():
    flops = [np.array(7.06396968e+10), np.array(1.44022375e+11), np.array(2.42780298e+11), np.array(3.66913477e+11)]
    accuracy_values = [np.array([68.1474785 , 66.86897278, 66.55477285]), np.array([68.89136434, 68.97262534, 67.66067346]), np.array([70.24774353, 69.6485877 , 69.03163393]), np.array([69.07265186, 69.29402153, 66.99118813])]
    labels = ['w1', 'w1.5', 'w2', 'w2.5']

    return flops, accuracy_values, labels