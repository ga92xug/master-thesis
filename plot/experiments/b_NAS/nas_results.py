import torch
import sys
import os
sys.path.append(f"{os.getcwd()}")
from plot.experiments.b_NAS.plot_nas_results import scalar_mappable
from experiments.b_NAS.common_best_architecture.util import *
from experiments.b_NAS.util import *
from plot.util import *
from plot.plot_functions import *


def plot_scalar_mappable(ax_client, save_folder_name, dataset_name):
    fig = scalar_mappable(ax_client.experiment)
    save_plot(
        figure=fig,
        name=dataset_name,
        folder_name=save_folder_name,
    )



def main():
    cfg, save_folder_name = plot_init("b_NAS/nas_results/")

    final_experiment_names = ["isic2019_4", "galaxy10_weighted_folder_2", "cifar10_2.2", "mnist_rot_2.2"]
    datasets = ["isic2019", "galaxy10", "cifar10", "mnist_rot"]

    for experiment_name, dataset_name in zip(final_experiment_names, datasets):
        client = get_ax_client_from_folder(folder = experiment_name)
        plot_scalar_mappable(client, save_folder_name, dataset_name)
        



if __name__ == "__main__":
    main()