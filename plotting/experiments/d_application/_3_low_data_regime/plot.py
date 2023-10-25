import os 
import sys
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


sys.path.append(os.getcwd())
from plotting.util import get_fig_size, save_plot

def example_data():
    # X-axis values - dataset reduction points in percentages
    reduction_points = np.array([100, 50])

    # Y-axis values - the test accuracy for different models at those reduction points
    efficientnet_acc = np.array([[95.6, 98.7], [80.3, 79.5]])
    vit_acc = np.array([[94.2, 93.8], [76.5, 77.2]])
    resnet_acc = np.array([[92.9, 93.1], [74.5, 75.3]])
    mobilenet_acc = np.array([[91.2, 92.3], [69.8, 70.1]])

    metrics_dict = {
        'EfficientNet': efficientnet_acc,
        'ViT': vit_acc,
        'ResNet': resnet_acc,
        'MobileNet': mobilenet_acc
    }

    return reduction_points, metrics_dict

from matplotlib.lines import Line2D

from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_metric_vs_reduction(
        metrics_dict: Dict[str, np.ndarray],
        metric_name: str,
        errorbar: bool = False,
        fig_size: Tuple[int, int] = (8, 4),
    ):
    plt.figure(figsize=fig_size)
    custom_lines = []
    reduction_points_longest = None
    
    for model, values in metrics_dict.items():
        metric_values = values.values()
        reduction_points = list(values.keys())

        if reduction_points_longest is None:
            reduction_points_longest = reduction_points
        elif len(reduction_points) > len(reduction_points_longest):
            reduction_points_longest = reduction_points

        means = [np.mean(values) for values in metric_values]
        stds = [np.std(values) for values in metric_values]
        
        if errorbar:
            errorbar_container = plt.errorbar(reduction_points, means, yerr=stds, label=model, marker='o', capsize=5, linestyle='-')
            line_color = errorbar_container[0].get_color()
        else:
            line, = plt.plot(reduction_points, means, label=model, marker='o', linestyle='-')
            line_color = line.get_color()
        
        custom_lines.append(Line2D([0], [0], color=line_color, marker='o', markersize=8, linestyle='-', linewidth=2))
    
    plt.xlabel('Dataset Size (%)')
    plt.ylabel(f'{metric_name}')
    plt.title(f'{metric_name} vs. Dataset Size')
    plt.xticks(reduction_points_longest)
    plt.grid(True)
    
    plt.legend(handles=custom_lines, title='Models', labels=list(metrics_dict.keys()))
    
    return plt.gcf()
    #save_plot(plt, name=f"example_{metric_name}_vs_dataset_size", folder_name='plotting/figures/d_application/3_low_data_regime')


def main():
    reduction_points, metrics_dict = example_data()
    plot_metric_vs_reduction(reduction_points, metrics_dict, 'Test Accuracy')
    plt.show()

if __name__ == '__main__':
    main()


