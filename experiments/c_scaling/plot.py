from math import log, sqrt
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def create_subplot(ax, flops, accuracy, xlabel, ylabel, labels=None):
    """Creates a single subplot.

    Args:
    ax: The axes object to plot on.
    flops: A list of FLOPs values.
    accuracy: A list of accuracy values.
    xlabel: The x-axis label.
    ylabel: The y-axis label.
    labels: A list of labels for the points (optional).
    """

    ax.plot(flops, accuracy, 'o-', markersize=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the x-axis limits to be slightly larger than the range of FLOPs values.
    ax.set_xlim(np.min(flops) - 0.5, np.max(flops) + 1)

    # Set the y-axis limits to be slightly larger than the range of accuracy values.
    ax.set_ylim(np.min(accuracy) - 0.1, np.max(accuracy) + 0.1)

    # Label individual points
    if labels is not None:
        for label, x, y in zip(labels, flops, accuracy):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(+5, -20), ha='center')

def scaling_individual(
        flops: List[float], 
        accuracy: List[float], 
        xlabel: str = "GFLOPs", 
        ylabel: str = "ISIC 2019 Valid Acc (%)",
        point_labels: List[str] = None  # Add a new argument for point labels
    ):
    """Creates all three subplots.

    Args:
    flops: A list of FLOPs values.
    accuracy: A list of accuracy values.
    xlabel: A list of x-axis labels.
    ylabel: A list of y-axis labels.
    point_labels: A list of labels for the points (optional).
    """
    # Create a figure object.
    fig = plt.figure(figsize=(9, 3))

    # Create a subplot grid.
    axarr = fig.subplots(1, 3)

    # Create each subplot.
    for i in range(3):
        if i == 0:
            create_subplot(axarr[i], flops[i], accuracy[i], xlabel, ylabel, point_labels[i] if point_labels else None)
        else:
            create_subplot(axarr[i], flops[i], accuracy[i], xlabel, "", point_labels[i] if point_labels else None)

    # Adjust the subplot spacing.
    fig.tight_layout()

def scaling_compound(
        flops_lists: List[List[float]],
        accuracy_lists: List[List[float]],
        xlabel: str = "GFLOPs",
        ylabel: str = "ISIC 2019 Valid Acc (%)",
        legend_labels: List[str] = None
    ):
    fig, ax = plt.subplots(figsize=(3, 3))

    for i, (flops, accuracy) in enumerate(zip(flops_lists, accuracy_lists)):
        color = plt.cm.tab10(i)
        ax.plot(flops, accuracy, color=color, marker='o', linestyle='-', markersize=5)

    if legend_labels:
        plt.legend(legend_labels, loc='lower right')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.grid(True)
    plt.tight_layout()


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
    new_point = sequence[-1] + log(sequence[-1], i * common_ratio) / common_ratio
    sequence.append(new_point)
  return sequence