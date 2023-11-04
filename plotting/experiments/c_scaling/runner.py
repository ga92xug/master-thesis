from omegaconf import DictConfig, OmegaConf
import os
import sys

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.c_scaling.plotting_functions import *
from plotting.experiments.c_scaling.wandb_data import *
from plotting.experiments.util import *

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
        )
    return kwargs


def produce_all_scaling_plots(
        cfg: DictConfig,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
        save_folder_name: str,
        xlabel: str = "GFLOPs",
        ylabel: str = "ISIC 2019 Valid Acc (%)",
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
    cfg, save_folder_name = plot_init("c_scaling/individual/", override=True)
    save_folder_name = f"{save_folder_name}_paper"
    wandb_entity = cfg.wandb.entity
    wandb_projects = "SL-Scaling"
    metric = "valid.acc_weighted"

    produce_all_scaling_plots(
        cfg=cfg,
        wandb_entity=wandb_entity,
        wandb_projects=wandb_projects,
        metric=metric,
        save_folder_name=save_folder_name, 
        combined_plot=False,
    )




if __name__ == "__main__":
    main()