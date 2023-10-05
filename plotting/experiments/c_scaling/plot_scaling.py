from math import log, sqrt
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def create_subplot(
        ax, 
        flops: List[float], 
        accuracy_values: List[float], 
        labels: List[str], 
        xlabel: str, 
        ylabel: str, 
        has_error_bars=True, 
        color='b',
        ylim_percentage: float = 10.0,
        xlim_percentage: float = 10.0,
        connect_dots: bool = True,
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
        if i > 0 and connect_dots:
            ax.plot([flops[i - 1], flops[i]], [np.mean(accuracy_values[i - 1]), np.mean(acc)], 'o-', lw=0.5, color=color)

    # Set the y-axis limits based on the overall range of accuracy values
    ylim_min = max(0, y_min - (ylim_percentage / 100) * (y_max - y_min)) # Set the lower limit to 0
    ylim_max = min(100, y_max + (ylim_percentage / 100) * (y_max - y_min)) # Set the upper limit to 100
    xlim_min = max(0, np.min(flops) - (xlim_percentage / 100) * (np.max(flops) - np.min(flops))) # Set the lower limit to 0
    xlim_max = np.max(flops) + (xlim_percentage / 100) * (np.max(flops) - np.min(flops))
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlim(xlim_min, xlim_max)

    # Label individual points
    yloc = 10
    for label, x, y in zip(labels, flops, [np.mean(acc) for acc in accuracy_values]):
        if label == labels[-1]:
            yloc *= -1
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(yloc, -15), ha='center')


def plot_scaling_individual(
        flops: List[float], 
        accuracy: List[float], 
        point_labels: List[str],  
        xlabel: str = "GFLOPs", 
        ylabel: str = "ISIC 2019 Valid Acc (%)",
        sharey: bool = False,
    ):
    """Creates all three subplots.

    Args:
    flops: A list of FLOPs values.
    accuracy: A list of accuracy values.
    point_labels: A list of labels for the points.
    xlabel: A list of x-axis labels.
    ylabel: A list of y-axis labels.
    """
    # Create a figure object.
    fig = plt.figure(figsize=(9, 3))

    # Create a subplot grid.
    axarr = fig.subplots(1, 3, sharey=sharey)

    # Create each subplot.
    for i in range(3):
        if i != 0:
            ylabel = ""

        create_subplot(
            ax=axarr[i],
            flops=flops[i],
            accuracy_values=accuracy[i],
            labels=point_labels[i],
            xlabel=xlabel,
            ylabel=ylabel,
            connect_dots= True if i != 1 else False,
        )

    # share y-axis

    # Adjust the subplot spacing.
    fig.tight_layout()

    return fig

def plot_scaling_compound_baseline(
        flops_lists: List[List[float]],
        accuracy_lists: List[List[float]],
        xlabel: str = "GFLOPs",
        ylabel: str = "ISIC 2019 Valid Acc (%)",
        legend_labels: List[str] = None
    ):
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figure size as needed

    # Calculate the final accuracy for each plot
    final_accs = [max(accuracy) for accuracy in accuracy_lists]

    # Sort the plots based on final accuracy in descending order
    sorted_indices = sorted(range(len(final_accs)), key=lambda i: final_accs[i], reverse=True)

    # Define linestyles for different plots based on their final accuracy
    linestyles = ['-', '--', '-.', ':']  # You can extend this list for more line styles

    for i, idx in enumerate(sorted_indices):
        flops = flops_lists[idx]
        accuracy = accuracy_lists[idx]
        linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles
        color = plt.cm.tab10(i)
        ax.plot(flops, accuracy, color=color, marker='o', linestyle=linestyle, markersize=5)

    if legend_labels:
        plt.legend([legend_labels[i] for i in sorted_indices], loc='lower right')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()

    return fig

def generate_fake_log_increasing_acc(start_value, common_ratio, n_terms):
    """Generates fake_log_increasing_acc

    Args:
        start_value: The first term in the sequence.
        common_ratio: The common ratio between terms in the sequence.
        n_terms: The number of terms in the sequence.

    Returns:
        A list of the terms in the sequence.
    """

    sequence = [start_value]
    for i in range(1, n_terms):
        new_point = sequence[-1] + log(sequence[-1], i * common_ratio) / common_ratio + np.random.normal(0, 1)
        sequence.append(new_point)
    return sequence


if __name__ == "__main__":
    legend = ["b3_d1, r96", "b4_d1, r128"]

    flops = [np.array(7.06396968e+10), np.array(1.44022375e+11), np.array(2.42780298e+11), np.array(3.66913477e+11)]
    accuracy_values = [np.array([68.1474785 , 66.86897278, 66.55477285]), np.array([68.89136434, 68.97262534, 67.66067346]), np.array([70.24774353, 69.6485877 , 69.03163393]), np.array([69.07265186, 69.29402153, 66.99118813])]
    accuracy_values = [np.mean(acc, axis=0) for acc in accuracy_values]

    flops2 = np.array([110, 221]) * 1e9
    accuracy_values2 = [[69.1, 72.29], [68.41, 70.78, 67.91]]
    accuracy_values2 = [np.mean(acc, axis=0) for acc in accuracy_values2]

    flops_list = [flops, flops2]
    accuracy_list = [accuracy_values, accuracy_values2]

    print("flops_list", flops_list)
    print("accuracy_list", accuracy_list)

    fig = plot_scaling_compound_baseline(
        flops_lists=flops_list,
        accuracy_lists=accuracy_list,
        legend_labels=legend,
    )

    # save plot
    fig.savefig("plotting/figures/c_scaling/compound/v1.png", dpi=300, bbox_inches='tight')