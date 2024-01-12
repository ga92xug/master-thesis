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


import sys
import os
sys.path.append(f"{os.getcwd()}")
from experiments.b_NAS.util import get_name_performance_metric
from experiments.b_NAS.plot import interact_contour_plotly, plot_marginal_effects
from plot.experiments.b_NAS.plot_nas_results import scalar_mappable

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


if __name__ == "__main__":
    # 6jqa39pv
    

    evaluate(ax_client=None, step=0, filepath="data/mnist_rot_2.2/ax_client.json")