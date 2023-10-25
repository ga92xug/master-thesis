from math import log, sqrt
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from plotting.util import get_fig_size, save_plot

def single_path_individual_scaling_plot(
        ax, 
        flops: List[float], 
        accuracy_values: List[float], 
        labels: List[str], 
        xlabel: str, 
        ylabel: str, 
        fig_size: tuple,
        has_error_bars: bool, 
        color: str = 'b',
        linestyle: str = '-',
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
    # some weird bug last entry in accuracy_values is all previous entries 
    new_accuracy_values = []
    for acc in accuracy_values:
        try:
            accuracy_mean = np.mean(acc)
            new_accuracy_values.append(acc)
        except Exception as e:
            print("acc", acc)
            print("e", e)
            continue

    accuracy_values = new_accuracy_values

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    handle = None
    for i, acc in enumerate(accuracy_values):
        if len(acc) > 1 and has_error_bars:
            # Calculate the mean and standard deviation for accuracy values
            #print("acc", acc)
            accuracy_mean = np.mean(acc)            

            accuracy_stddev = np.std(acc)
            ax.errorbar(flops[i], accuracy_mean, yerr=accuracy_stddev, fmt='o-', markersize=5, color=color)

        else:
            ax.plot(flops[i], acc, 'o-', markersize=5, color=color)

        # Connect the current data point to the previous one with a line
        if i > 0 and connect_dots:
            handle, = ax.plot(
                flops, 
                [np.mean(_acc) for _acc in accuracy_values], 
                linestyle=linestyle, 
                lw=0.5, 
                color=color, 
                label=labels[i] if labels else None
            )

    # Label individual points
    if labels is not None:
        standard_fig_size = (6.4, 4.8)
        # adjust the label location based on the fig size
        xloc_org = 15
        yloc_org = -15


        for label, x, y in zip(labels, flops, [np.mean(acc) for acc in accuracy_values]):
            xloc = xloc_org + (len(labels[0]))
            if label == labels[-1]:
                # set the label of the last to be on the left instead of right
                xloc *= -1

            xloc = xloc / standard_fig_size[0] * fig_size[0]
            yloc = yloc_org / standard_fig_size[1] * fig_size[1]
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(xloc, yloc), ha='center')

    return handle


def multipath_individual_scaling_plot(
        flops_lists: List[List[float]], 
        accuracy_values_lists: List[List[List[float]]], 
        colors: List[str], 
        linestyles: List[str],
        xlabel: str = "FLOPs", 
        ylabel: str = "Weigthed Validation Accuracy",
        labels_lists: List[List[str]] = None,
        legend_labels: List[str] = None,
        ax=None,
        has_error_bars=True,
        connect_dots: bool = True,
        fig_size: tuple = (6.4, 4.8),
        save_name: str = None,
        save_folder_name: str = None,
    ):
    """
    Creates a single plot with the possibility of multiple paths.
    Serves as a wrapper for single_path_individual_scaling_plot.
    """

    assert isinstance(colors, list), "colors must be a list"
    assert isinstance(linestyles, list), "linestyles must be a list"
    assert len(flops_lists) == len(accuracy_values_lists), "flops_lists and accuracy_values_lists must have the same length"
    assert len(flops_lists) <= len(colors), "colors must be at least as long as flops_lists"
    assert len(flops_lists) <= len(linestyles), "linestyles must be at least as long as flops_lists"
    

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = None

    legend_handles = []
    for i, (flops, accuracy_values) in enumerate(
            zip(flops_lists, accuracy_values_lists)
        ):
        labels = get_list(labels_lists, i, None)
        color = colors[i]
        linestyle = linestyles[i]

        assert len(flops) == len(accuracy_values), "flops and accuracy_values must have the same length"

        handle = single_path_individual_scaling_plot(
            ax=ax,
            flops=flops,
            accuracy_values=accuracy_values,
            labels=labels,
            xlabel=xlabel,
            ylabel=ylabel,
            has_error_bars=has_error_bars,
            color=color,
            linestyle=linestyle,
            connect_dots=connect_dots,
            fig_size=fig_size,
        )
        legend_handles.append(handle)

    if legend_labels is not None:
        ax.legend(legend_handles, legend_labels, loc='best')


    if fig is not None:
        save_plot(
            figure=fig,
            name=save_name,
            folder_name=save_folder_name,
        )


def combined_individual_scaling(
        combined_plot_dict: dict, 
        xlabel: str, 
        ylabel: str,
        save_folder_name: str, 
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
    fig_size = (9, 3)
    #fig_size = get_fig_size(fig_size)

    # Create a figure object.
    fig = plt.figure(figsize=fig_size)

    # Create a subplot grid.
    axarr = fig.subplots(1, 3, sharey=sharey)

    # Create each subplot.
    for i, (exp_name, values) in enumerate(combined_plot_dict.items()):
        if i != 0:
            ylabel = ""

        fig_size_individual = (fig_size[0] / 3, fig_size[1])
        multipath_individual_scaling_plot(
            ax=axarr[i],
            xlabel=xlabel,
            ylabel=ylabel,
            fig_size=fig_size_individual,
            **values
        )

    fig.tight_layout()

    save_plot(
        figure=fig,
        name="combined_width_depth_res_scaling",
        folder_name=save_folder_name,
    )

    return fig

def get_list(
        list_object: List, 
        index: int, 
        default_value=None
    ):
    """
    Implements the get function for lists.
    """
    if list_object is not None and len(list_object) > index:
        result = list_object[index]
    else:
        result = default_value

    return result

def plot_scaling_compound_baseline(
        flops_lists: List[List[float]],
        accuracy_lists: List[List[float]],
        xlabel: str = "GFLOPs",
        ylabel: str = "ISIC 2019 Valid Acc Weighted (%)",
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

    #print("flops_list", flops_list)
    #print("accuracy_list", accuracy_list)

    fig = plot_scaling_compound_baseline(
        flops_lists=flops_list,
        accuracy_lists=accuracy_list,
        legend_labels=legend,
    )

    # save plot
    fig.savefig("plotting/figures/c_scaling/compound/v1.png", dpi=300, bbox_inches='tight')