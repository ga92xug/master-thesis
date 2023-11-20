from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from numpy import save, shape
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
import sys

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.plotting_utils import *


METRIC_2_YLABEL = {
    "valid.acc": "Validation Accuracy [%]",
    "valid.acc_weighted": "Validation Accuracy Weighted [%]",
    "test.acc": "Test Accuracy [%]",
    "test.acc_weighted": "Test Accuracy Weighted [%]",
}

def metric2label(metric: str):
    return METRIC_2_YLABEL[metric]

def get_short_labels(labels: List[str], short_labels_function: str):
    if short_labels_function is None:
        return labels
    else:
        transform_label = eval(f"lambda s: {short_labels_function}")
        return [transform_label(label) for label in labels]


def plot_histogram(
        ax, 
        data,
        mode: str = "param_count", # ["param_count", "flops"]
        short_labels_function: str = None,
        use_color_palette: bool = False,
    ):
    assert mode in ["param_count", "flops"], f"Mode {mode} is not supported!"

    if use_color_palette:
        colors = [label2color(label) for label in list(data.keys())]
    else:
        colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

    data = get_metric_from_downloaded_data(data, mode)
    short_labels_function = get_short_labels(list(data.keys()), short_labels_function)

    data = list(data.values())

    if mode == "param_count":
        data = [d / 1e6 for d in data]
    else:
        data = [d / 1e9 for d in data]
    
    x = np.arange(len(short_labels_function))  # Create an array of evenly spaced x values
    ax.bar(x, data, color=colors)

    ylabel = mode.replace("_", " ")
    ylabel = f"FLOPs [$10^9$]" if ylabel == "flops" else "Parameters [$10^6$]"
    ax.set_ylabel(ylabel)

    ax.set_xticks(x)  # Set the x-ticks to the evenly spaced values
    ax.set_xticklabels(short_labels_function, rotation=90, ha='center')  # Rotate labels by 45 degrees and align to the right


def plot_validation_accuracy(
        ax : plt.Axes, 
        downloaded_data: Dict[str, Dict[str, Dict[str, List]]], 
        metric : str,
        ylim_percentage: float = 10.0,
        horizontal_line: dict = None, 
        window_size: int = 3,
        use_color_palette: bool = False,
    ):
    """
    Plot the validation accuracy for all labels and runs. \
    The data is smoothed with a moving average. \
    

    Parameters:
    - ax (plt.Axes): The axes to plot on.
    - downloaded_data (dict): Nested dictionary containing data (Label: Run ID: Metric: List[float]).
    - metric (str): The name of the metric to plot.
    - ylim_percentage (float): The percentage to add to the y-axis limits.
    - horizontal_line (dict): If there is a horizontal line to plot.
    - window_size (int): The window size for smoothing the data.
    """

    transformed_data = transform_data_to_arrays(downloaded_data, metric, window_size)

    min_acc = np.inf
    max_acc = -np.inf    
    for label, runs_data in transformed_data.items():
        color = label2color(label) if use_color_palette else None

        if runs_data.shape[0] > 1:
            # Multiple runs for this label
            # Plot the mean and standard deviation
            mean = np.mean(runs_data, axis=0)
            std = np.std(runs_data, axis=0)

            
            ax.plot(mean, label=label, linewidth=1, color=color)
            ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color=color)

            min_acc = min(min_acc, mean.min())
            max_acc = max(max_acc, mean.max())
        else:
            # Only one run for this label
            # Plot the single run
            ax.plot(runs_data[0], label=label, linewidth=1, color=color)

            min_acc = min(min_acc, runs_data[0].min())
            max_acc = max(max_acc, runs_data[0].max())


    if horizontal_line is not None:
        label = horizontal_line['label']
        line_y = horizontal_line['y']
        color = horizontal_line['color']
        ax.axhline(y=line_y, color=color, linestyle=horizontal_line['linestyle'])    
        ax.annotate(label, xy=(0, line_y), xytext=(5, line_y + 1), textcoords='offset points', color=color)


    ax.set_xlabel('Epoch')
    ax.set_ylabel(METRIC_2_YLABEL[metric])
    ylim_min = max(0, min_acc - (ylim_percentage / 100) * (max_acc - min_acc)) # Set the lower limit to 0
    ylim_max = min(100, max_acc + (ylim_percentage / 100) * (max_acc - min_acc)) # Set the upper limit to 100
    # ax.set_ylim(ylim_min, ylim_max)
    # ax.set_xlim(0, len(runs_data[0]) - 1)

    # Set the legend labels to upper case
    h, legends = ax.get_legend_handles_labels()
    legends = [l[0].upper() + l[1:] for l in legends]
    ax.legend(h, legends, loc='lower right')
    ax.grid(True)


def get_metric_from_downloaded_data(
        downloaded_data: Dict[str, Dict[str, Dict[str, List]]],
        metric: str, 
        aggregation_func: str ="equal",
        smoothing_window_size: int = 1,
        **kwargs,
    )-> Dict[str, np.ndarray]:
    """
    Extract a specific metric from the downloaded data. \
    The information about the run_id is lost. \
    Aggregates the data for one metric into a numpy array.

    Parameters:
    - downloaded_data (dict): Nested dictionary containing data (Label: Run ID: Metric: List[float]).
    - metric (str): The specific metric to extract.
    - smoothing_window_size (int): The window size for smoothing the data. 1 for no smoothing.

    Returns:
    - extracted_metric (dict): Extracted metric data.
    """

    extracted_metric = {}  # Store aggregated run data

    # Iterate through labels and runs
    for label, runs_data in downloaded_data.items():
        extracted_metric[label] = []  # Initialize the label entry
        for run_id, run_data in runs_data.items():
            if smoothing_window_size == 1:
                extracted_metric[label].append(run_data[metric]) 
            else:
                extracted_metric[label].append(smooth_data(run_data[metric], smoothing_window_size))

        if aggregation_func == "equal":
            # If the aggregation function is "equal", the metric should be the same for all runs
            assert np.allclose(extracted_metric[label][0], extracted_metric[label][1:], **kwargs), \
                f"Metric {metric} is not equal for all runs of label {label}!, {extracted_metric[label]}"
            
            extracted_metric[label] = np.array(extracted_metric[label][0])

        elif aggregation_func == "mean":
            extracted_metric[label] = np.mean(extracted_metric[label], axis=0)

        elif aggregation_func == "no_aggregation":
            extracted_metric[label] = np.array(extracted_metric[label])

        elif aggregation_func == "extract_last":
            extracted_metric[label] = np.array(extracted_metric[label])[:, -1]
        else:
            NotImplementedError(f"Aggregation function {aggregation_func} is not supported!")

    return extracted_metric


def transform_data_to_arrays(
        downloaded_data: Dict[str, Dict[str, Dict[str, List]]], 
        # Label: Run ID: Metric: List[float] 
        metric: str, 
        window_size: int = 3
    ) -> Dict[str, np.ndarray]:
    """
    Aggregates the data for one metric into a numpy array. \
    The information about the run_id is lost. \
    The data is smoothed with a moving average. \
    Returns a dictionary with the labels as keys and the aggregated data as values.
    """

    transformed_data = {} 

    # Iterate through labels and runs
    for label, runs_data in downloaded_data.items():
        transformed_data[label] = [] 
        for run_id, run_data in runs_data.items():
            if window_size == 1:
                smoothed_data = run_data[metric]
            else:
                smoothed_data = smooth_data(run_data[metric], window_size)
            transformed_data[label].append(smoothed_data)  

        transformed_data[label] = np.array(transformed_data[label])  # Convert the list to a numpy array

    return transformed_data


def create_combined_plot(
        downloaded_data: Dict[str, Dict[str, Dict[str, List]]], 
        metric: str,
        short_labels: str = None,
        fig_size: Tuple = (8, 4.5),
        use_color_palette: bool = False,
        **kwargs,
    ) -> plt.Figure:
    """
    Create a combined plot of flops, total parameters, and validation accuracy. \
    The data is smoothed with a moving average. \
    
    Parameters:
    - downloaded_data (dict): Nested dictionary containing data (Label: Run ID: Metric: List[float]).
    - metric (str): The specific metric to extract.
    - **kwargs: Additional arguments for the validation acc about plotting SOTA lines.
    """

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1])

    ax2 = plt.subplot(gs[0])
    plot_validation_accuracy(ax2, downloaded_data, metric, use_color_palette=use_color_palette, **kwargs)
    
    ax1 = plt.subplot(gs[1])
    plot_histogram(ax1, downloaded_data, short_labels_function=short_labels, use_color_palette=use_color_palette, mode="flops")

    ax3 = plt.subplot(gs[2])
    plot_histogram(ax3, downloaded_data, short_labels_function=short_labels, use_color_palette=use_color_palette, mode="param_count")
    
    plt.tight_layout()
    plt.show()
    return fig
