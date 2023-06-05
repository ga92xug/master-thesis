import wandb
from typing import List, Optional
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

from ax.service.utils.report_utils import exp_to_df
from ax.service.ax_client import AxClient
from ax.modelbridge.factory import get_MOO_NEHVI 

# Plotting imports and initialization
from ax.plot.contour import interact_contour_plotly
from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly

# resume wandb run
# wandb.init(
#     project="scaling-laws-nas_high_level", 
#     entity="ga92xug", 
#     resume="allow",
#     id="v0p3z512",  # resume the run using the saved run ID
# )

def evaluate(
        ax_client = None,
        filepath: str = None,
):
    assert ax_client is not None or filepath is not None, "Either ax_client or filepath must be provided"
    # load ax client
    if ax_client is None:
        ax_client = AxClient.load_from_json_file(filepath=filepath)

    experiment = ax_client.experiment
    data = experiment.fetch_data()

    # pareto frontier
    pareto_frontier = _pareto_frontier_scatter_2d_plotly(experiment)
    pareto_frontier
    wandb.log({"pareto_frontier": wandb.Plotly(pareto_frontier)}, step=50, commit=True)

    # scalar mappable
    fig = scalar_mappable(ax_client.experiment, title="Equivariant NAS on MNIST-rot", new_point=[0.958, 86], new_point_label="eq_wrn_16_4")
    fig.show()
    wandb.log({"pareto_frontier_image": wandb.Image(fig)}, step=50)


    # contour plots
    model = get_MOO_NEHVI(
        experiment=experiment, 
        data=data,
    )
    valid_acc_interact_contour_plotly = interact_contour_plotly(model, metric_name="valid_acc", lower_is_better=False)
    gflops_interact_contour_plotly = interact_contour_plotly(model, metric_name="gflops", lower_is_better=True)
    wandb.log({"valid_acc_contour": wandb.Plotly(valid_acc_interact_contour_plotly)}, step=50, commit=True)
    wandb.log({"gflops_contour": wandb.Plotly(gflops_interact_contour_plotly)}, step=50, commit=True)




def scalar_mappable(
        experiment,
        title: str = "Equivariant NAS on MNIST-rot",
        new_point: Optional[List[float]] = None,
        new_point_label: Optional[str] = None,
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

    # Convert experiment data to DataFrame and sort by trial_index
    df = exp_to_df(experiment).sort_values(by=["trial_index"])
    
    # Extract required data columns
    outcomes = df[["valid_acc", "gflops"]].values

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
    axes.set_ylabel("gflops")

    # Add a new point if given
    if new_point:
        new_point = np.array([new_point])
        sc_new = axes.scatter(new_point[:, 0], new_point[:, 1], c='red', label=new_point_label if new_point_label else 'eq_wrn_16_4')

        # Update the legend
        handles, labels = axes.get_legend_handles_labels()
        handles = [h for h, l in zip(handles, labels) if l != (new_point_label if new_point_label else 'eq_wrn_16_4')]
        labels = [l for l in labels if l != (new_point_label if new_point_label else 'eq_wrn_16_4')]
        handles.append(sc_new)
        labels.append(new_point_label if new_point_label else 'eq_wrn_16_4')
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