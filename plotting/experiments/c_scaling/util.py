from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def baseline_and_scaling_exp_2_scaling_exp(data):
    """Transforms the input data into the format required for create_subplot.

    Args:
    data: A dictionary containing accuracy and FLOPs data.

    Returns:
    transformed_data: A dictionary with transformed data.
    """
    transformed_data = {}

    keys = list(data.keys())
    assert len(keys) == 2, f"Expected two keys, got {len(keys)}"
    for key in keys:
        if "baseline" in key:
            baseline_key = key
        
    if baseline_key is None:
        raise ValueError("No baseline key found.")
    
    # other key is the scaling key
    scaling_key = [key for key in keys if key != baseline_key][0]

    # weight labels
    scaling_label = scaling_key[0]

    # Add the baseline data
    transformed_data[scaling_label + "1"] = data[baseline_key]["all"]

    for key, values in data[scaling_key].items():
        transformed_data[scaling_label + str(key)] = values

    return transformed_data



def transform_data_for_plot(data):
    """Transforms the input data into the format required for create_subplot.

    Args:
    data: A dictionary containing accuracy and FLOPs data.

    Returns:
    flops: A list of FLOPs values.
    accuracy_values: A list of lists containing accuracy values for each data point.
    labels: A list of labels for the points.
    """

    data = dict(sorted(data.items()))

    flops = []
    accuracy_values = []
    labels = []

    for label, values in data.items():
        flops.append(values['flops'])
        accuracy_values.append(values['valid.acc_weighted'])
        labels.append(label)

    return flops, accuracy_values, labels

def create_subplot(
        ax, 
        flops: List[float], 
        accuracy_values: List[float], 
        xlabel: str, 
        ylabel: str, 
        labels=None, 
        has_error_bars=True, 
        color='b',
        ylim_percentage: float = 10.0,
        xlim_percentage: float = 10.0,
    ):
    """Creates a single subplot with optional error bars for accuracy values.

    Args:
    ax: The axes object to plot on.
    flops: A list of FLOPs values.
    accuracy_values: A list of lists containing accuracy values for each data point.
    xlabel: The x-axis label.
    ylabel: The y-axis label.
    labels: A list of labels for the points (optional).
    has_error_bars: A boolean indicating whether to plot error bars.
    color: The color for points and connecting lines (e.g., 'b' for blue).
    """
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the x-axis limits to be slightly larger than the range of FLOPs values.
    #ax.set_xlim(np.min(flops) * 0.9, np.max(flops) * 1.1)

    y_min, y_max = float('inf'), float('-inf')  # Initialize y-axis limits

    for i, acc in enumerate(accuracy_values):
        if len(acc) > 1 and has_error_bars:
            # Calculate the mean and standard deviation for accuracy values
            accuracy_mean = np.mean(acc)
            accuracy_stddev = np.std(acc)
            ax.errorbar(flops[i], accuracy_mean, yerr=accuracy_stddev, fmt='o-', markersize=5, color=color)

            # Update y-axis limits based on the current data point with error bars
            y_min = min(y_min, accuracy_mean - accuracy_stddev)
            y_max = max(y_max, accuracy_mean + accuracy_stddev)
        else:
            ax.plot(flops[i], acc, 'o-', markersize=5, color=color)

            # Update y-axis limits based on the current data point without error bars
            y_min = min(y_min, min(acc))
            y_max = max(y_max, max(acc))

        # Connect the current data point to the previous one with a line
        if i > 0:
            ax.plot([flops[i - 1], flops[i]], [np.mean(accuracy_values[i - 1]), np.mean(acc)], 'o-', lw=0.5, color=color)

    # Set the y-axis limits based on the overall range of accuracy values
    ylim_min = max(0, y_min - (ylim_percentage / 100) * (y_max - y_min)) # Set the lower limit to 0
    ylim_max = min(100, y_max + (ylim_percentage / 100) * (y_max - y_min)) # Set the upper limit to 100
    xlim_min = max(0, np.min(flops) - (xlim_percentage / 100) * (np.max(flops) - np.min(flops))) # Set the lower limit to 0
    xlim_max = np.max(flops) + (xlim_percentage / 100) * (np.max(flops) - np.min(flops))
    #ax.set_ylim(ylim_min, ylim_max)
    #ax.set_xlim(xlim_min, xlim_max)

    # Label individual points
    yloc = 10
    if labels is not None:
        for label, x, y in zip(labels, flops, [np.mean(acc) for acc in accuracy_values]):
            if label == labels[-1]:
                yloc *= -1
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(yloc, -15), ha='center')
