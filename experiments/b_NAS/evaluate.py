import re
import torch
import wandb
from typing import List, Optional
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

from ax.service.utils.report_utils import exp_to_df
from ax.service.ax_client import AxClient
from ax.modelbridge.factory import get_MOO_NEHVI 

# Plotting imports and initialization
#from ax.plot.contour import interact_contour_plotly
from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly
from ax.plot.feature_importances import plot_feature_importance_by_feature_plotly
from plot import interact_contour_plotly, plot_marginal_effects

METADATA = {
        "mnist_rot": {
            "name": "MNIST-rot",
            "point": [0.9885, 157],
            "label": "eq_wrn_16_4",
        },
        "cifar10": {
            "name": "CIFAR10",
            "point": [0.9125, 226],
            "label": "eq_wrn_16_4",
        },
        "galaxy10": {
            "name": "Galaxy10",
            "point": [0.8235, 6265],
            "label": "eq_wrn_16_4",
        },
        "unkown": {
            "name": "Unknown",
            "point": False,
            "label": None,
        },
    }

def evaluate(
        ax_client: AxClient,
        step: int,
        filepath: str = None,
):
    assert ax_client is not None or filepath is not None, "Either ax_client or filepath must be provided"
    # load ax client
    if ax_client is None:
        ax_client = AxClient.load_from_json_file(filepath=filepath)

    experiment = ax_client.experiment
    data = experiment.fetch_data()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # pareto frontier
    pareto_frontier = _pareto_frontier_scatter_2d_plotly(experiment)
    pareto_frontier
    wandb.log({"pareto_frontier": wandb.Plotly(pareto_frontier)}, step=step, commit=True)
    step += 1

    # scalar mappable
    fig = scalar_mappable(ax_client.experiment)
    wandb.log({"pareto_frontier_image": wandb.Image(fig)}, step=step, commit=True)
    step += 1

    # contour plots
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = get_MOO_NEHVI(
        experiment=experiment, 
        data=data,
        device=device
    )
    valid_acc_contour_plot, gflops_contour_plot = get_contour_plots(
        model=model,
    )
    wandb.log({"valid_acc_contour": wandb.Plotly(valid_acc_contour_plot)}, step=step, commit=True)
    step += 1
    wandb.log({"gflops_contour": wandb.Plotly(gflops_contour_plot)}, step=step, commit=True)
    step += 1

    # feature importance
    feature_importance = plot_feature_importance_by_feature_plotly(model)
    wandb.log({"feature_importance": wandb.Plotly(feature_importance)}, step=step, commit=True)
    step += 1

    # marginal effects
    for metric in ["valid_acc_weighted", "gflops"]:
        marginal_effects = plot_marginal_effects(model, metric)
        for idx, fig in enumerate(marginal_effects):
            wandb.log({f"{metric}_{idx}_marginal_effects": wandb.Plotly(fig)}, step=step, commit=True)
            step += 1


def get_contour_plots(
        model,
        density: int = 10,
):
    valid_acc_interact_contour_plotly = interact_contour_plotly(model, metric_name="valid_acc_weighted", lower_is_better=False, density=density)
    gflops_interact_contour_plotly = interact_contour_plotly(model, metric_name="gflops", lower_is_better=True, density=density)
    return valid_acc_interact_contour_plotly, gflops_interact_contour_plotly

def scalar_mappable(
        experiment,
        title: str = None,
        baseline_point: Optional[List[float]] = None,
        baseline_label: Optional[str] = None,
        ):
    """
    This function creates a scatter plot of an experiment's data sorted by trial_index.
    The points are color-coded based on their iteration (trial_index). An optional point can be added and highlighted.

    :param experiment: A data object representing the experiment. Needs to be convertible to a DataFrame
                       with 'valid_acc', 'gflops', and 'trial_index' columns.
    :param title: A string to use as the plot title. Default is "Equivariant NAS on MNIST-rot".
    :param new_point: A list of two floats representing the 'valid_acc' and 'gflops' of the new point. 
                      If None, no new point is added. Default is None.
    :param new_point_label: A string representing the label of the new point. 
                            If None, 'eq_wrn_16_4' is used. Default is None.
    """
    
    meta_data = get_meta_information(experiment.name)
    name, point, label = meta_data["name"], meta_data["point"], meta_data["label"]

    title = title if title else f"Equivariant NAS on {name}"
    baseline_point = baseline_point if baseline_point else point
    baseline_label = baseline_label if baseline_label else label

    # Convert experiment data to DataFrame and sort by trial_index
    df = exp_to_df(experiment).sort_values(by=["trial_index"])
    
    # Extract required data columns
    outcomes = df[["valid_acc_weighted", "gflops"]].values

    # Create figure and axes for the plot
    fig, axes = plt.subplots(1, 1, figsize=(8,6))
    
    train_obj = outcomes
    
    # Color map for the scatter plot
    cm = plt.cm.get_cmap('viridis')

    # Extract batch_number from the DataFrame
    trial_index_values = df.trial_index.values
    
    # Create scatter plot
    sc = axes.scatter(train_obj[:, 0], train_obj[:,1], c=trial_index_values, alpha=0.8)
    axes.set_title(title)
    axes.set_xlabel("valid acc")
    axes.set_ylabel("GFLOPs")

    # Add a new point if given
    if baseline_point:
        baseline_point = np.array([baseline_point])
        sc_new = axes.scatter(baseline_point[:, 0], baseline_point[:, 1], c='red', label=baseline_label if baseline_label else 'eq_wrn_16_4')

        # Update the legend
        handles, labels = axes.get_legend_handles_labels()
        handles = [h for h, l in zip(handles, labels) if l != (baseline_label if baseline_label else 'eq_wrn_16_4')]
        labels = [l for l in labels if l != (baseline_label if baseline_label else 'eq_wrn_16_4')]
        handles.append(sc_new)
        labels.append(baseline_label if baseline_label else 'eq_wrn_16_4')
        axes.legend(handles, labels)

    # Normalize the color bar
    norm = plt.Normalize(trial_index_values.min(), trial_index_values.max())
    sm =  ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    fig.subplots_adjust(right=0.9)
    
    # Create a color bar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
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
    pattern = r'(mnist_rot|cifar10|galaxy10)'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return None

if __name__ == "__main__":
    # 6jqa39pv
    

    evaluate(ax_client=None, step=0, filepath="data/mnist_rot_2.2/ax_client.json")