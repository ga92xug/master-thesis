import os
from re import L
import sys
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from numpy import save, shape
import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from matplotlib import gridspec
sys.path.append('../scaling-laws-ecnn') # add parent directory

#from training.speed_test import main

title_size = 12
label_size = 10


def smooth_data(data, w: int = 3):
    return np.convolve(data, np.ones(w), 'valid') / w

def save_plot(figure, name, folder_name='figures/first_experiments'):
    name = name.replace(' ', '_').lower()

    if not os.path.exists(f'{folder_name}'):
        os.makedirs(f'{folder_name}')

    figure.savefig(f'{folder_name}/{name}.png', dpi=300, bbox_inches = "tight")


def plot_flops(ax, downloaded_data):
    data = get_metric_from_downloaded_data(downloaded_data, "flops")
    labels = list(data.keys())
    flops = list(data.values())

    colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    x = np.arange(len(labels))  # Create an array of evenly spaced x values
    ax.bar(x, flops, color=colors)
    ax.set_ylabel('FLOPs')

    ax.set_xticks(x)  # Set the x-ticks to the evenly spaced values
    ax.set_xticklabels(labels, rotation=90, ha='center')  # Rotate labels by 45 degrees and align to the right


def plot_total_parameters(ax, downloaded_data):
    data = get_metric_from_downloaded_data(downloaded_data, "param_count")
    labels = list(data.keys())
    total_params = list(data.values())

    colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    x = np.arange(len(labels))  # Create an array of evenly spaced x values
    ax.bar(x, total_params, color=colors)
    ax.set_ylabel('Param count')

    ax.set_xticks(x)  # Set the x-ticks to the evenly spaced values
    ax.set_xticklabels(labels, rotation=90, ha='center')  # Rotate labels by 45 degrees and align to the right


def plot_validation_accuracy2(
        ax : plt.Axes, 
        downloaded_data: Dict[str, Dict[str, Dict[str, List]]], 
        # Label: Run ID: Metric: List[float]
        metric : str,
        ylim_percentage: float = 10.0,
        horizontal_line: dict = None, 
        window_size: int = 3,
    ):
    metric2ylabel = {
        "valid.acc": "Validation accuracy",
        "valid.acc_weighted": "Weighted validation accuracy",
    }

    #max_acc = max([max(acc) for acc in valid_accs])
    #min_acc = min([min(acc) for acc in valid_accs])
    #
    #ylim_min = min_acc - (ylim_percentage / 100) * (max_acc - min_acc)
    #ylim_max = max_acc + (ylim_percentage / 100) * (max_acc - min_acc)
    transformed_data = transform_data_to_arrays(downloaded_data, metric, window_size)

    min_acc = np.inf
    max_acc = -np.inf    
    for label, runs_data in transformed_data.items():
        if runs_data.shape[0] > 1:
            # Multiple runs for this label
            # Plot the mean and standard deviation
            mean = np.mean(runs_data, axis=0)
            std = np.std(runs_data, axis=0)
        
            ax.plot(mean, label=label, linewidth=1)
            ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

            min_acc = min(min_acc, mean.min())
            max_acc = max(max_acc, mean.max())
        else:
            # Only one run for this label
            # Plot the single run
            ax.plot(runs_data[0], label=label, linewidth=1)

            min_acc = min(min_acc, runs_data[0].min())
            max_acc = max(max_acc, runs_data[0].max())


    if horizontal_line is not None:
        label = horizontal_line['label']
        line_y = horizontal_line['y']
        color = horizontal_line['color']
        ax.axhline(y=line_y, color=color, linestyle=horizontal_line['linestyle'])    
        ax.annotate(label, xy=(0, line_y), xytext=(5, line_y + 1), textcoords='offset points', fontsize=8, color=color)


    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric2ylabel[metric])
    ylim_min = min_acc - (ylim_percentage / 100) * (max_acc - min_acc)
    ylim_max = max_acc + (ylim_percentage / 100) * (max_acc - min_acc)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlim(0, len(runs_data[0]) - 1)
    ax.legend()
    ax.grid(True)


def get_metric_from_downloaded_data(downloaded_data, metric, aggregation_func="equal"):
    """
    Extract a specific metric from the downloaded data.

    Parameters:
    - downloaded_data (dict): Nested dictionary containing data.
    - metric (str): The specific metric to extract.

    Returns:
    - extracted_metric (dict): Extracted metric data.
    """

    extracted_metric = {}  # Dictionary to store the extracted metric data

    # Iterate through labels and runs
    for label, runs_data in downloaded_data.items():
        extracted_metric[label] = []  # Initialize the label entry
        for run_id, run_data in runs_data.items():
            extracted_metric[label].append(run_data[metric])  # Append the metric data to the label entry

        if aggregation_func == "equal":
            # If the aggregation function is "equal", the metric should be the same for all runs
            assert np.allclose(extracted_metric[label][0], extracted_metric[label][1:]), \
                f"Metric {metric} is not equal for all runs of label {label}!"
            
            extracted_metric[label] = extracted_metric[label][0]  # Extract the metric from the list
        
        else:
            NotImplementedError(f"Aggregation function {aggregation_func} is not supported!")

    return extracted_metric


def transform_data_to_arrays(
        downloaded_data: Dict[str, Dict[str, Dict[str, List]]], 
        # Label: Run ID: Metric: List[float] 
        metric: str, 
        window_size: int = 3
    ) -> Dict[str, np.ndarray]:
    transformed_data = {}  # Dictionary to store the transformed data

    # Iterate through labels and runs
    for label, runs_data in downloaded_data.items():
        transformed_data[label] = [] 
        for run_id, run_data in runs_data.items():
            smoothed_data = smooth_data(run_data[metric], window_size)
            transformed_data[label].append(smoothed_data)  

        transformed_data[label] = np.array(transformed_data[label])  # Convert the list to a numpy array

    return transformed_data


def create_combined_plot2(
        downloaded_data: Dict,
        metric: str,
        **kwargs,
    ) -> plt.Figure:

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1])

    ax2 = plt.subplot(gs[0])
    plot_validation_accuracy2(ax2, downloaded_data, metric, **kwargs)
    
    ax1 = plt.subplot(gs[1])
    plot_flops(ax1, downloaded_data)

    ax3 = plt.subplot(gs[2])
    plot_total_parameters(ax3, downloaded_data)
    

    #fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    #save_plot(fig, "valid_acc_flops_combined", title)
    plt.show()

    return fig

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

