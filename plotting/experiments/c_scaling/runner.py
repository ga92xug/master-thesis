import pickle
from omegaconf import DictConfig, OmegaConf
import os
import sys


sys.path.append(f"{os.getcwd()}")
from plotting.experiments.c_scaling.plotting_functions import *
from plotting.experiments.c_scaling.wandb_data import *
from plotting.experiments.plotting_utils import *

def scaling_plot_data(
        paths2filter_dict: dict,
        exp_dict: dict,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
        xlabel: str,
        ylabel: str,
        save_folder_name: str,
        name: str,    
        produce_plot: bool = True,
    ):
    """
    
    Args:
    - sub_exp_dict: a sub_exp is a sub experiment that is part of a larger experiment.\
        For example baseline_3blocks is a subexperiment of the width_scaling experiment.

    """

    wandb_data = get_data_for_exp(paths2filter_dict, wandb_entity, wandb_projects, metric)
    flops, accuracy_values, labels, legend_labels = transform_data(wandb_data, metric, exp_dict, name)

    save_data(
        flops=flops,
        accuracy_values=accuracy_values,
        labels=labels,
        legend_labels=legend_labels,
        save_folder_name=save_folder_name,
        name=name,
    )    

    colors = exp_dict["colors"]
    linestyles = exp_dict["linestyles"]

    kwargs = {
        "flops_lists": flops,
        "accuracy_values_lists": accuracy_values,
        "colors": colors,
        "linestyles": linestyles,
        "labels_lists": labels if name != "compound_scaling" else None,
        "legend_labels": legend_labels # if name == "compound_scaling" else None,
    }

    
    if produce_plot:
        multipath_individual_scaling_plot(
            **kwargs,
            xlabel=xlabel,
            ylabel=ylabel,
            save_name = name,
            save_folder_name=save_folder_name,
            fig_size=FIG_SIZE,
        )
    return kwargs


def produce_all_scaling_plots(
        cfg: DictConfig,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
        save_folder_name: str,
        xlabel: str = "FLOPs [$10^9$]",
        ylabel: str = "Validation Accuracy Weighted [%]",
        combined_plot: bool = False, 
    ):
    """
    exp_dict: dict
        paths: dict
            label_mask: path_name
        colors: list
        linestyles: list

    filter_dict: dict
        path_name: dict
            filters: dict
            group_by: str or list
    """

    filter_dict = OmegaConf.to_container(
        cfg.experiments.filter_groupby_dict, resolve=True, throw_on_missing=True)

    exp2labels = OmegaConf.to_container(
        cfg.experiments.plots, resolve=True, throw_on_missing=True)

    combined_plot_dict = {}

    for exp_name, exp_dict in exp2labels.items():
        #if exp_name != "compound_scaling":
        #    continue
        print(exp_name)

        # extract all sub experiments that belong to one experiment
        paths2filter_dict = {}
        for label_mask, path_name in exp_dict["paths"].items():
            paths2filter_dict[path_name] = filter_dict[path_name]


        save_dict = scaling_plot_data(
            paths2filter_dict=paths2filter_dict,
            exp_dict=exp_dict,
            wandb_entity=wandb_entity,
            wandb_projects=wandb_projects,
            metric=metric,
            save_folder_name=save_folder_name,  
            name=exp_name,  
            xlabel=xlabel,
            ylabel=ylabel,
        )

        if combined_plot and exp_name != "compound_scaling":
            combined_plot_dict[exp_name] = save_dict

    
    if combined_plot:
        print("combined_plot")
        combined_individual_scaling(
            combined_plot_dict,
            xlabel=xlabel,
            ylabel=ylabel,
            save_folder_name=save_folder_name,
            sharey=True,
        )


def main():
    cfg, save_folder_name = plot_init("c_scaling/individual", override=True)
    save_folder_name = f"{save_folder_name}_new"
    wandb_entity = cfg.wandb_entity
    wandb_projects = "SL-Scaling"
    metric = "valid.acc_weighted"

    produce_all_scaling_plots(
        cfg=cfg,
        wandb_entity=wandb_entity,
        wandb_projects=wandb_projects,
        metric=metric,
        save_folder_name=save_folder_name, 
        combined_plot=True,
    )


def save_data(
        flops: list,
        accuracy_values: list,
        labels: list,
        legend_labels: list,
        save_folder_name: str,
        name: str,
):
    if legend_labels is None:
        copy_legend_labels = ["3 Blocks"]
    else:
        copy_legend_labels = legend_labels.copy()

    pickel = {}
    for i, legend in enumerate(copy_legend_labels):
        pickel[legend] = {}
        for j, label in enumerate(labels[i]):
            pickel[legend][label] = {}
            pickel[legend][label]["flops"] = float(flops[i][j])
            pickel[legend][label]["accuracy"] = accuracy_values[i][j].tolist()

    save_path = f"{save_folder_name}/{name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(pickel, f)

    # open the file and check if the data is correct
    with open(save_path, "rb") as f:
        data = pickle.load(f)
        print(data)


if __name__ == "__main__":
    main()