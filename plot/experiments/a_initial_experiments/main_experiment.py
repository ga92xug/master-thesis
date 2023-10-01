import os
import sys
import numpy as np
from typing import List, Union
from omegaconf import OmegaConf
from sympy import O
sys.path.append(f"{os.getcwd()}")
from plot.util import *
from plot.plot_functions import *


def add_flops(downloaded_data, GFLOPs):
    i = 0
    for label, runs_data in downloaded_data.items():
        for run_id, run_data in runs_data.items():
            run_data["flops"] = GFLOPs[i] * 1e9
            print(f"Added to {label} {run_id} gflops: {GFLOPs[i]}")
            i += 1
    return downloaded_data


def plot(
        dataset2metric: Dict[str, str],
        wandb_entity: str,
        wandb_projects: str,
        save_folder_name: str,
        save_name: str,
        labels_run_ids: Dict[str, Dict[str, Union[List[str], str]]],
        dataset: str,  
        GFLOPs: List[float] = None,
        short_labels: str = None,
        **kwargs,
    ):
    metric = dataset2metric[dataset]

    downloaded_data = download_data(
        entity=wandb_entity, 
        projects=wandb_projects, 
        labels_run_ids=labels_run_ids,
        metric=metric,
        name_param_count="total_parameters"
    )
    if GFLOPs is not None:
        downloaded_data = add_flops(
            downloaded_data=downloaded_data,
            GFLOPs=GFLOPs,
        )

    fig_size = get_fig_size((8,4))
    fig = create_combined_plot(
        downloaded_data=downloaded_data,
        metric=metric,
        fig_size=fig_size,
        short_labels=short_labels,
        **kwargs,
    )

    save_plot(
        figure=fig,
        name=save_name,
        folder_name=save_folder_name,
    )
    

def main():
    cfg, save_folder_name = plot_init("a_initial_experiments/main_experiment/", override=True)
    wandb_entity = cfg.wandb.entity
    wandb_projects = ["SL-first-experiments"]

    experiments_2_run_ids = OmegaConf.to_container(
            cfg.experiments, resolve=True, throw_on_missing=True)
    dataset2metric = OmegaConf.to_container(
            cfg.dataset2metric, resolve=True, throw_on_missing=True)

    for name, run_info in experiments_2_run_ids.items():
        #if name == "mnist_rot":
        #    continue
        print(name)
        
        plot(
            dataset2metric=dataset2metric,
            wandb_entity=wandb_entity, 
            wandb_projects=wandb_projects, 
            save_folder_name=save_folder_name,
            save_name=name,
            dataset="cifar10",
            **run_info
        )


if __name__ == "__main__":
    main()