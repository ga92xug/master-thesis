from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
import sys
from adjustText import adjust_text

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
    
def group_labels(unique_label:str):
    if unique_label == "C":
        return "Cyclic"
    elif unique_label == "D":
        return "Dihedral"
    else:
        raise ValueError("Unknown group")


def flops2location(
        data: Dict,
        bubble_sizes: List[float],
        max_params: float,
        ax: plt.Axes,
    ):
    flops = [model['flops'] for model in data.values()]
    max_flops = max(flops)
    mean_flops = np.mean(flops)
    accs = [model['acc'] for model in data.values()]

    accs_range = max(accs) - min(accs)

    sizes = [(params_size / max_params) * 1000 for params_size in bubble_sizes]
    print("Sizes", sizes)
    for i, params_size in enumerate(bubble_sizes):
        size = (params_size / max_params) * 1000
        #if i > 0:
        print(i, bubble_sizes[i:], np.sum(bubble_sizes[i:]), max_flops, max_flops - np.sum(bubble_sizes[i:]))
        flop_location = max_flops - np.sum(sizes[i:]) 
        print(flop_location)
        ax.scatter(flop_location, min(accs) + accs_range * 0.1, s=size, c='grey', alpha=0.5)
        ax.text(flop_location, min(accs) + accs_range * 0.2, f'{params_size/1e6} [$10^6$]')


def plot_combined_acc_flops_params(
        data: Dict,
        metric_name: str,
        bubble_sizes: List[float] = [1, 2, 4, 8],
        max_params: float = 7,
        bubble_mode: str = "legend",
        split_labels_on: str = "s[0]",
    ):
    #max_params = max(model['param_count'] for model in data.values())
    print("max_params", max_params)

    fig, ax = plt.subplots()
    texts = []

    list_params = [model['param_count'] for model in data.values()]
    print("list_params", list_params)
    list_flops = [model['flops'] for model in data.values()]
    print("list_flops", list_flops)
    list_acc = [model['acc'] for model in data.values()]
    print("list_acc", list_acc)
    list_size = [(params_size / max_params) * 1000 for params_size in list_params]
    print("list_size", list_size)

    if split_labels_on is None:
        sc = ax.scatter(list_flops, list_acc, s=list_size)
    else:
        labels = list(data.keys())
        # eval(f"lambda s: {short_labels_function}")
        transform_label_function = eval(f"lambda s: {split_labels_on}")
        transformed_labels = [transform_label_function(label) for label in labels]

        unique_labels = list(set(transformed_labels))
        print("unique_labels", unique_labels)
        #colors = plt.cm.get_cmap("tab10", len(unique_labels))
        colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

        list_scatters = []
        for i, unique_label in enumerate(unique_labels):
            indices = [i for i, label in enumerate(labels) if unique_label in label]
            print("indices", indices)
            x = [list_flops[i] for i in indices]
            y = [list_acc[i] for i in indices]
            s = [list_size[i] for i in indices]
            print("x", x)
            print("y", y)
            print("s", s)


            sc = ax.scatter(x, y, s=s, color=colors[i], label=group_labels(unique_label))
            list_scatters.append(sc)

        first_legend = plt.legend(handles=list_scatters, loc='lower right', title="Group")
        ax.add_artist(first_legend)

    for model_name, model in data.items():
        size = (model['param_count'] / max_params) * 1000
        #ax.scatter(model['flops'], model['acc'], s=size)
        texts.append(plt.text(model['flops'], model['acc'], model_name, ha='center', va='center'))

    # Improve the placement of the text labels to reduce overlaps
    adjust_text(texts, expand_objects=(1.05, 3), expand_points=(1.05, 3))

    # Bubble size legend
    if bubble_mode == "legend":
        #plt.legend(*sc.legend_elements("sizes", num=6))
        list_scatters = []
        for params_size in bubble_sizes:
            size = (params_size / max_params) * 1000
            sc = ax.scatter([], [], s=size, c='grey', alpha=0.5, label=f'{params_size}')
            list_scatters.append(sc)

        # ncol=len(df.columns)
        leg = ax.legend(
            handles=list_scatters,
            loc='center left', 
            borderpad=1, 
            labelspacing=1, 
            bbox_to_anchor=(1, 0.5), 
            title="Parameters\n[$10^6$]"
        )
        leg.get_title().set_multialignment('center')
    elif bubble_mode == "text":
        flops2location(data, bubble_sizes, max_params, ax)

    ax.grid(True)
    if max(list_flops) - min(list_flops) > 2:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
    if max(list_acc) - min(list_acc) > 2:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
    ax.set_xlabel('FLOPs [$10^9$]')
    ax.set_ylabel(metric_name)
    ax.margins(0.1, 0.1)
    #ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')
    return fig


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
        transformed_data[label] = {}
        for i, (run_id, run_data) in enumerate(runs_data.items()):
            if window_size == 1:
                smoothed_data = run_data[metric]
            else:
                smoothed_data = smooth_data(run_data[metric], window_size)


            transformed_data[label]["acc"] = smoothed_data[-1]
            if i == 0:
                # "param_count", "flops"
                transformed_data[label]["flops"] = run_data["flops"] / 1e9
                transformed_data[label]["param_count"] = run_data["param_count"] / 1e6


        transformed_data[label]["acc"] = np.mean(transformed_data[label]["acc"])  # Convert the list to a numpy array

    return transformed_data


def create_combined_plot(
        downloaded_data: Dict[str, Dict[str, Dict[str, List]]], 
        metric: str,
        split_labels_on: str = None,
        fig_size: Tuple = (8, 6),
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

    transformed_data = transform_data_to_arrays(downloaded_data, metric, 3)
    metric_name = metric2label(metric)
    fig = plot_combined_acc_flops_params(
        transformed_data, 
        metric_name=metric_name,
        split_labels_on=split_labels_on,
    )
    
    return fig


if __name__ == "__main__":
    # Load
    models = {
        "AlexNet": {"accuracy": 56.5, "FLOPs": 0.72, "params": 60e6},
        "VGG-16": {"accuracy": 71.5, "FLOPs": 15.5, "params": 138e6},
        "ResNet-50": {"accuracy": 76.2, "FLOPs": 3.8, "params": 25.6e6},
        "Inception-v3": {"accuracy": 78.8, "FLOPs": 5.0, "params": 23.8e6},
        "NASNet-A-Large": {"accuracy": 82.7, "FLOPs": 23.8, "params": 88.9e6},
    }

    fig = plot_combined_acc_flops_params(models)

    # Save
    fig.savefig("plotting/figures/combined_acc_flops_params.png", bbox_inches='tight')
