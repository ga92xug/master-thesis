import re
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

from ax.service.utils.report_utils import exp_to_df

import sys
import os

sys.path.append(f"{os.getcwd()}")
from experiments.b_NAS.util import get_name_performance_metric
from plotting.experiments.plotting_utils import get_fig_size, label2color

# textwidth latex 5.78853in
# textwidth in cm: \printinunitsof{in}\prntlen{\textwidth}

METADATA = {
        "mnist_rot": {
            "name": "MNIST-rot",
            "point": {"location": [0.9915000200271606, 313.543363446], "label": "Eq-WRN-16-4"},
            "batch_size": 64,
        },
        "cifar10": {
            "name": "CIFAR10",
            "point": {"location": [0.921999990940094, 540.934433542], "label": "Eq-WRN-16-4"},
            "batch_size": 128,
        },
        "galaxy10": {
            "name": "Galaxy10",
            "point": {"location": [0.8116401433944702, 6144.474499862], "label": "Eq-WRN-16-4"},
            "batch_size": 128,
        },
        "isic2019": {
            "name": "ISIC2019",
            "point": {"location": [0.5097538232803345, 6144.474499862], "label": "Eq-WRN-16-4"},
            # https://paperswithcode.com/sota/classification-on-isic-2019
            #"line": {"x": 0.6519, "label": "SOTA", "color": "red", "linestyle": "dashed"},
            "batch_size": 128,
        },
        "unkown": {
            "name": "Unknown",
            "point": False,
            "label": None,
        },
    }
    

def scalar_mappable(
        experiment,
        title: str = None,
        fig_size: tuple = (7, 3.5),
        reduction: float = 0.49,
    ):
    """
    This function creates a scatter plot of an experiment's data sorted by trial_index.
    The points are color-coded based on their iteration (trial_index). An optional point can be added and highlighted.
    """
    #fig_size = get_fig_size(fig_size, reduction=reduction)
    
    meta_data = get_meta_information(experiment.name)
    name = meta_data["name"]
    batch_size = meta_data["batch_size"]

    title = title if title else f"Equivariant NAS on {name}"

    # Convert experiment data to DataFrame and sort by trial_index
    df = exp_to_df(experiment).sort_values(by=["trial_index"])
    
    # Extract required data columns
    name_performance_metric = get_name_performance_metric(experiment)
    df["gflops"] = df["gflops"] / batch_size
    df[name_performance_metric] = df[name_performance_metric] * 100
    outcomes = df[[name_performance_metric, "gflops"]].values

    # Create figure and axes for the plot
    fig, axes = plt.subplots(1, 1, figsize=fig_size)
    
    train_obj = outcomes
    
    # Color map for the scatter plot
    cm = plt.cm.get_cmap('BuPu')

    # Extract batch_number from the DataFrame
    trial_index_values = df.trial_index.values
    
    # Create scatter plot
    sc = axes.scatter(train_obj[:, 0], train_obj[:,1], c=trial_index_values, alpha=0.8, cmap=cm)
    #axes.set_title(title)

    
    xlabel = "Validation Accuracy Weighted [%]" if name_performance_metric == "valid_acc_weighted" else "Validation Accuracy [%]"
    axes.set_xlabel(xlabel)
    axes.set_ylabel("FLOPs [$10^9$]")

    # Add a new point if given
    baseline_point = meta_data["point"]["location"]
    baseline_label = meta_data["point"]["label"]
    if baseline_point:
        color = label2color(baseline_label)
        baseline_point = np.array([baseline_point[0] * 100, baseline_point[1] / batch_size]) 
        sc_new = axes.scatter(baseline_point[0], baseline_point[1], c=color)

        # Add description close to the label of the point
        axes.text(baseline_point[0] - 0.02, baseline_point[1] - 0.09, baseline_label, color=color, va='top', ha='right')

    # Add a line if given
    line = meta_data.get("line", None)
    if line is not None:
        label = line['label']
        line_x = line['x']
        color = line['color']
        
        # Add a horizontal line
        axes.axvline(x=line_x, color=color, linestyle=line['linestyle'])    
        
        # Add description close to the line label
        axes.text(line_x - 0.01, baseline_point[:, 1], label, color=color, va='top', ha='right')

    # Normalize the color bar
    norm = plt.Normalize(trial_index_values.min(), trial_index_values.max())
    sm =  ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    fig.subplots_adjust(right=0.9)
    
    # Create a color bar
    # Get the position (Bbox) of the axis
    axis_position = axes.get_position()

    # Extract the height from the position
    axis_height = axis_position.y1 - axis_position.y0
    cbar_ax = fig.add_axes([0.92, axis_position.y0, 0.04, axis_height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title("Iteration")

    return fig

def get_meta_information(name: str):
    result = match_substring(name)
    if result:
        return METADATA[result]
    else:
        return METADATA["unkown"]
    
def match_substring(string):
    pattern = f'({"|".join(METADATA.keys())})'
    #pattern = r'(mnist_rot|cifar10|galaxy10|isic2019)'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return None

