import os
from re import L
import sys
from typing import List, Optional
import matplotlib.pyplot as plt
from numpy import save
import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from matplotlib import gridspec
sys.path.append('../scaling-laws-ecnn') # add parent directory

#from training.speed_test import main

title_size = 12
label_size = 10

def plot_model_data(model_data, vs_param):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
    fig.suptitle(f'Scaling {vs_param}', fontsize=14, fontweight='bold')
    
    for model_name, data in model_data.items():
        vs = data[:, 0]
        param_count = data[:, 1]
        model_building_time = data[:, 2]
        train_time = data[:, 3]
        gflops = data[:, 4]
        
        ax[0].plot(vs, param_count, label=model_name)
        ax[1].plot(vs, model_building_time, label=model_name)
        ax[2].plot(vs, train_time, label=model_name)
        ax[3].plot(vs, gflops, label=model_name)
    
    ax[0].set_xlabel(f'{vs_param}')
    ax[0].set_ylabel('Parameter Count')
    ax[0].set_title(f'Parameter Count')
    ax[0].legend()

    ax[1].set_xlabel(f'{vs_param}')
    ax[1].set_ylabel('Model Building Time')
    ax[1].set_title(f'Model Building Time')
    ax[1].legend()

    ax[2].set_xlabel(f'{vs_param}')
    ax[2].set_ylabel('Train Time')
    ax[2].set_title(f'Train Time')
    ax[2].legend()

    ax[3].set_xlabel(f'{vs_param}')
    ax[3].set_ylabel('GFLOPs')
    ax[3].set_title(f'GFLOPs')
    ax[3].legend()

    plt.tight_layout()
    plt.savefig(f'scaling/figures/{model_name.split("_")[0]}_{vs_param}_scaling.png')


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


def download_data(
        entity, 
        projects, 
        run_ids, 
        name_param_count : str = "param_count", 
        metric : str = "valid.acc",
    ):    
    valid_accs = []
    param_counts = []
    flops = []
    
    for run_id in run_ids:
        run = find_run(entity, projects, run_id)
        
        valid_accs.append(run.history(keys=[metric]).values[:, 1])
        param_counts.append(run.history(keys=[name_param_count]).values[:, 1][0])
        
        # some of the runs do not have flops
        try:
            flops.append(run.history(keys=['GFLOPs']).values[:, 1][0] * 1e9)
        except:
            pass

    return valid_accs, param_counts, flops


def smooth_data(x, w: int = 3):
    return np.convolve(x, np.ones(w), 'valid') / w

def save_plot(figure, name, folder_name='figures/first_experiments'):
    name = name.replace(' ', '_').lower()

    if not os.path.exists(f'{folder_name}'):
        os.makedirs(f'{folder_name}')

    figure.savefig(f'{folder_name}/{name}.png', dpi=300, bbox_inches = "tight")


def plot_flops(ax, gflops, labels):
    colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    x = np.arange(len(labels))  # Create an array of evenly spaced x values
    ax.bar(x, gflops, color=colors)
    ax.set_ylabel('FLOPs')

    ax.set_xticks(x)  # Set the x-ticks to the evenly spaced values
    ax.set_xticklabels(labels, rotation=90, ha='center')  # Rotate labels by 45 degrees and align to the right


def plot_total_parameters(ax, total_params, labels):
    colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    x = np.arange(len(labels))  # Create an array of evenly spaced x values
    ax.bar(x, total_params, color=colors)
    ax.set_ylabel('Param count')

    ax.set_xticks(x)  # Set the x-ticks to the evenly spaced values
    ax.set_xticklabels(labels, rotation=90, ha='center')  # Rotate labels by 45 degrees and align to the right


def plot_validation_accuracy(
        ax : plt.Axes, 
        valid_accs : List[float], 
        labels : List[str], 
        metric : str,
        ylim_percentage: float = 10.0,
        horizontal_line: dict = None, 
    ):
    metric2ylabel = {
        "valid.acc": "Validation accuracy",
        "valid.acc_weighted": "Weighted validation accuracy",
    }

    max_acc = max([max(acc) for acc in valid_accs])
    min_acc = min([min(acc) for acc in valid_accs])
    
    ylim_min = min_acc - (ylim_percentage / 100) * (max_acc - min_acc)
    ylim_max = max_acc + (ylim_percentage / 100) * (max_acc - min_acc)

    for i, acc in enumerate(valid_accs):
        acc = smooth_data(acc)
        ax.plot(acc, label=labels[i], linewidth=1) # 3 originally

    if horizontal_line is not None:
        label = horizontal_line['label']
        line_y = horizontal_line['y']
        color = horizontal_line['color']
        ax.axhline(y=line_y, color=color, linestyle=horizontal_line['linestyle'])    
        ax.annotate(label, xy=(0, line_y), xytext=(5, line_y + 1), textcoords='offset points', fontsize=8, color=color)


    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric2ylabel[metric])
    ax.set_ylim(ylim_min, ylim_max)
    ax.legend()
    ax.grid(True)
    #ax.set_yticks([round(ylim_min, 1), round(ylim_max, 1)]) 


def create_combined_plot(
        valid_accs: List[float], 
        flops: List[int], 
        total_params: List[int], 
        long_labels: List[str],
        metric: str,
        short_labels: Optional[List[str]] = None, 
        #title: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
    if short_labels is None:
        short_labels = long_labels

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1])

    ax2 = plt.subplot(gs[0])
    plot_validation_accuracy(ax2, valid_accs, long_labels, metric, **kwargs)
    
    ax1 = plt.subplot(gs[1])
    plot_flops(ax1, flops, short_labels)

    ax3 = plt.subplot(gs[2])
    plot_total_parameters(ax3, total_params, short_labels)
    

    #fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    #save_plot(fig, "valid_acc_flops_combined", title)
    plt.show()

    return fig

def binary_search_over_model_scaling(
        search_param, scale_param, scaling_factor, initial_range, constraint_func,
        overrides, global_overrides, max_iterations=50, tolerance=0.01):
    """
    does not work for now

    Perform a binary search to find the best scaling parameter for a given model configuration.

    Args:
        search_param (dict): A dictionary containing the parameter to search for (key) and its initial value (value).
        scale_param (dict): A dictionary containing the parameter to scale for (key) and its initial value (value).
        scaling_factor (float): The desired scaling factor for the scale_param.
        initial_range (tuple): A tuple containing the initial lower and upper bounds for the search_param.
        constraint_func (function): A function to apply model-specific constraints to the search_param.
        overrides (dict): A dictionary containing model configuration parameters that are not part of the search.
        global_overrides (dict): A dictionary containing global configuration parameters for the model.
        max_iterations (int, optional): The maximum number of iterations for the binary search. Default is 100.
        tolerance (float, optional): The acceptable error threshold for convergence. Default is 0.01.

    Returns:
        mid: The best scaling parameter found for the search_param.

    Raises:
        ValueError: If the search_param or scale_param is not valid or supported.
    """
    assert False, "not supported for now"

    scale_indices = {
        "param_count": 0,
        "model_building_time": 1,
        "train_time": 2
    }
    
    search_key = list(search_param.keys())[0]
    scale_key = list(scale_param.keys())[0]
    scale_index = scale_indices[scale_key]

    initial_value = search_param[search_key]
    local_overrides = {**overrides, **search_param}
    total_overrides = global_overrides.copy()
    total_overrides.update(local_overrides)

    base_stats = run(overrides=total_overrides, verbose=0)

    lower_bound, upper_bound = initial_range

    is_integer = isinstance(initial_value, int)

    iteration = 0
    while iteration < max_iterations:
        mid = (lower_bound + upper_bound) / 2
        if is_integer:
            mid = int(mid)

        mid = constraint_func(mid)
        local_overrides[search_key] = mid
        total_overrides.update(local_overrides)

        try:
            stats = run(overrides=total_overrides, verbose=0)
        except:
            upper_bound = mid * 0.9
            continue

        if abs(stats[scale_index] - base_stats[scale_index] * scaling_factor) <= tolerance:
            break
        elif stats[scale_index] > base_stats[scale_index] * scaling_factor:
            upper_bound = mid * 0.9
        else:
            lower_bound = mid * 1.1

        iteration += 1

    print(f"Best {search_key} found: {mid}")
    return mid

