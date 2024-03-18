import os 
import sys
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plotting.experiments.d_application.util import name2color, sorting_key

sys.path.append(os.getcwd())
from plotting.experiments.plotting_utils import *

def plot_low_data_regime(
        metrics_dict: Dict[str, np.ndarray],
        metric_name: str,
        fig_size: Tuple[int, int],
        errorbar: bool = False,
        reverse_order: bool = True,
    ):
    plt.figure(figsize=fig_size)
    custom_lines = []
    reduction_points_longest = None

    # order metrics_dict by key
    metrics_dict = dict(sorted(metrics_dict.items(), key=lambda item: sorting_key(item[0])))    
    
    for model, values in metrics_dict.items():
        color = label2color(model)
        metric_values = list(values.values())
        reduction_points = list(values.keys())
        if reverse_order:
            metric_values.reverse()
            reduction_points.reverse()

        reduction_points = [int((float(reduction_point) * 100)) for reduction_point in reduction_points]

        if reduction_points_longest is None:
            reduction_points_longest = reduction_points
        elif len(reduction_points) > len(reduction_points_longest):
            reduction_points_longest = reduction_points

        means = [np.mean(values) for values in metric_values]
        stds = [np.std(values) for values in metric_values]
        
        if errorbar:
            errorbar_container = plt.errorbar(reduction_points, means, yerr=stds, label=model, marker='o', capsize=5, linestyle='-', color=color)
            line_color = errorbar_container[0].get_color()
        else:
            line, = plt.plot(reduction_points, means, label=model, marker='o', linestyle='-', color=color)
            line_color = line.get_color()
        
        custom_lines.append(Line2D([0], [0], color=color, marker='o', markersize=8, linestyle='-', linewidth=2))
    
    plt.xlabel('Dataset Size [%]')
    metric_name = metric_name.replace(".", " ").replace("_", " ").replace("/", " ").replace("acc", "Accuracy")
    metric_name += " [%]"
    metric_name = metric_name[0].upper() + metric_name[1:]
    plt.ylabel(f'{metric_name}')
    #plt.title(f'{metric_name} vs. Dataset size')
    plt.grid(True)
    
    plt.legend(handles=custom_lines, labels=list(metrics_dict.keys()))
    return plt.gcf()
    

def example_data():
    """
    Some example data to test the plotting function.
    """
    metrics_dict = {
        'eq_nasnet': {'1': [73.50000143, 74.25000072, 73.75000119, 73.75000119, 73.25000167], '0.5': [74.75000024, 74.25000072, 72.50000238], '0.3': [74.75000024, 74.25000072, 73.00000191], '0.1': [74.25000072, 74.25000072, 72.25000262], '0.05': [73.25000167, 73.50000143, 72.75000215]}, 
        'vit_pre': {'1': [74.50000048, 74.25000072, 73.25000167, 74.25000072, 71.74999714], '0.3': [69.74999905, 70.99999785, 68.75      , 72.25000262, 72.75000215]}, 
        'vit': {'1': [64.74999785, 68.00000072, 69.24999952, 63.49999905, 69.24999952], '0.3': [70.24999857, 69.24999952, 71.74999714, 66.50000215, 64.24999833]}, 
        'efficientnet_pre': {'1': [71.49999738, 70.99999785, 72.00000286, 73.00000191, 72.25000262], '0.5': [65.24999738, 64.74999785, 66.75000191, 67.50000119, 65.24999738], '0.3': [72.00000286, 70.99999785, 70.99999785, 69.74999905, 70.74999809]}, 
        'efficientnet': {'1': [70.74999809, 77.49999762, 73.75000119, 74.25000072, 68.99999976], '0.3': [74.75000024, 71.74999714, 71.24999762, 70.99999785, 72.25000262]}
    }
    return metrics_dict


if __name__ == '__main__':
    metrics_dict = example_data()
    plot_low_data_regime(metrics_dict, "test.acc")
    save_plot(plt, name=f"example", folder_name='plotting/figures/d_application/low_data_regime')


