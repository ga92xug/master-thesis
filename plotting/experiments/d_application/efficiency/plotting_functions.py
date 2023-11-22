import os 
import sys
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plotting.experiments.plot_acc_flops_params import metric2label

sys.path.append(os.getcwd())
from plotting.experiments.d_application.util import name2color
from plotting.experiments.plotting_utils import *


def plot_model_efficiency_with_test_acc_histogram(
        model_data, 
        valid_metric_name="val_accuracy",
        test_metric_name="test_accuracy",
        title='Model Efficiency and Test Accuracy Comparison'
    ):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Plot FLOPs vs. Validation Accuracy
    for model_name, data in model_data.items():
        color = label2color(model_name)
        # Calculate mean and standard deviation for validation accuracy
        mean_val_acc = np.mean(data["valid_metric"], axis=0)
        std_val_acc = np.std(data["valid_metric"], axis=0)
        flops = data['FLOPs_4_val_values']

        # Plotting the mean validation accuracy
        line, = ax1.plot(flops, mean_val_acc, label=model_name, color=color)

        # Plotting the standard deviation for validation accuracy
        ax1.fill_between(flops, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, alpha=0.2, color=line.get_color())

    ax1.set_xlabel('FLOPs [$10^9$]')
    ylabel = metric2label(valid_metric_name)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    # Plot Mean Test Accuracy Histogram with Whiskers for Standard Deviation
    for i, model_name in enumerate(model_data.keys()):
        mean_test_acc = np.mean(model_data[model_name]["test_metric"])
        std_test_acc = np.std(model_data[model_name]["test_metric"])
        
        # Bar for mean test accuracy
        color = label2color(model_name)
        ax2.bar(i, mean_test_acc, color=color)

        # Whisker for standard deviation
        ax2.errorbar(i, mean_test_acc, yerr=std_test_acc, color='black', fmt='none')

    ax2.set_xticks(range(len(model_data)))
    ax2.set_xticklabels(model_data.keys())
    #ax2.set_xlabel('Model')
    ylabel = metric2label(test_metric_name)
    ax2.set_ylabel(ylabel)
    #ax2.set_title('Mean Test Accuracy per Model')

    plt.tight_layout()
    plt.show()
    return fig


def example_data():
    """
    Some example data to test the plotting function.
    """
    num_values_eq_nasnet = 4

    example_data = {
        'EfficientNet': {
            'flops': np.linspace(0, 100, 10),
            'val_accuracy': np.random.rand(5, 10),  # Random validation accuracy values for 5 runs
            'test_accuracy': np.random.rand(5)      # Random test accuracy values
        },
        'EQ-NASNet': {
            'flops': np.linspace(0, 100, num_values_eq_nasnet),
            'val_accuracy': np.random.rand(5, num_values_eq_nasnet) * 1.4,  # Higher validation accuracy for EQ-NASNet
            'test_accuracy': np.random.rand(5) * 1.4      # Higher test accuracy for EQ-NASNet
        }
    }
    return example_data


if __name__ == '__main__':
    metrics_dict = example_data()
    plot_model_efficiency_with_test_acc_histogram(metrics_dict)
    save_plot(plt, name=f"example", folder_name='plotting/figures/d_application/efficiency')


