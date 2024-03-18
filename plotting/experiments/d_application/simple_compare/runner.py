import os
import sys
import numpy as np
from typing import List, Union
from omegaconf import OmegaConf

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.wandb_utils import *
from plotting.experiments.plotting_utils import *
from plotting.experiments.plot_acc_flops_params_nicer import *

def manual_data():
    models = {
        "Gessert et al. single model": {"accuracy": 68.8, "params": 40.76e6, "FLOPs": 19.97e9},
        "EfficientNet pre-trained": {"accuracy": 62.97, "params": 4.01e6, "FLOPs": 0.14e9},
        "EfficientNet": {"accuracy": 39.76, "params": 4.01e6, "FLOPs": 0.14e9},
        "ViT pre-trained": {"accuracy": 66.34, "params": 85.69e6, "FLOPs": 5.5e9},
        "ViT": {"accuracy": 43.51, "params": 85.69e6, "FLOPs": 5.5e9},
        "Eq-NASNet": {"accuracy": 69.69, "params": 0.66e6, "FLOPs": 1.7e9}
    }


def plot(
        dataset2metric: Dict[str, str],
        wandb_entity: str,
        wandb_projects: str,
        save_folder_name: str,
        save_name: str,
        labels_run_ids: Dict[str, Dict[str, Union[List[str], str]]],
        dataset: str,  
        GFLOPs: List[float] = None,
        split_labels_on: str = None,
        **kwargs,
    ):
    metric = dataset2metric[dataset]

    downloaded_data = {
        "Gessert et al. single model": {"accuracy": 68.8, "params": 40.76e6, "FLOPs": 19.97e9},
        "EfficientNet pre-trained": {"accuracy": 62.97, "params": 4.01e6, "FLOPs": 0.14e9},
        "EfficientNet": {"accuracy": 39.76, "params": 4.01e6, "FLOPs": 0.14e9},
        "ViT pre-trained": {"accuracy": 66.34, "params": 85.69e6, "FLOPs": 5.5e9},
        "ViT": {"accuracy": 43.51, "params": 85.69e6, "FLOPs": 5.5e9},
        "Eq-NASNet": {"accuracy": 69.69, "params": 0.66e6, "FLOPs": 1.7e9}
    }


    print("split_labels_on", split_labels_on)
    fig = create_combined_plot(
        downloaded_data=downloaded_data,
        metric=metric,
        split_labels_on=split_labels_on,
        **kwargs,
    )

    save_plot(
        figure=fig,
        name=save_name,
        folder_name=save_folder_name,
    )
    

def main():
    cfg, save_folder_name = plot_init("a_initial_experiments/main_experiment", override=True)
    wandb_entity = cfg.wandb_entity
    wandb_projects = ["SL-first-experiments"]

    experiments_2_run_ids = OmegaConf.to_container(
            cfg.experiments, resolve=True, throw_on_missing=True)
    dataset2metric = OmegaConf.to_container(
            cfg.dataset2metric, resolve=True, throw_on_missing=True)

    for name, run_info in experiments_2_run_ids.items():
        if name != "cyclic_or_dihedral":
            continue
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