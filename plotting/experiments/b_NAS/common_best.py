import os
import sys
from typing import List, Union
from omegaconf import OmegaConf

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.plotting_utils import *
from plotting.experiments.plot_acc_flops_params_nicer import *
from plotting.experiments.wandb_utils import *


dataset2metric = {
    # normal
    "mnist12k": "valid.acc",
    "mnist_rot": "valid.acc",
    "cifar10": "valid.acc",
    "ciar10_rot": "valid.acc",

    # weighted
    "galaxy10": "valid.acc_weighted",
    "isic2019": "valid.acc_weighted",
} 

def plot(
        dataset2metric: Dict[str, str],
        wandb_entity: str,
        wandb_projects: str,
        save_folder_name: str,
        save_name: str,
        labels_run_ids: Dict[str, Dict[str, Union[List[str], str]]],
        dataset: str,  
        **kwargs,
    ):
    metric = dataset2metric[dataset]

    downloaded_data = get_wandb_data_multiple_runs(
        entity=wandb_entity, 
        projects=wandb_projects, 
        labels_run_ids=labels_run_ids,
        metric=metric,
    )


    fig = create_combined_plot(
        downloaded_data=downloaded_data,
        metric=metric,
        use_color_palette=True,
        **kwargs,
    )

    save_plot(
        figure=fig,
        name=save_name,
        folder_name=save_folder_name,
    )
    

def main():
    cfg, save_folder_name = plot_init("b_nas/common_best", override=True)
    wandb_entity = cfg.wandb_entity
    wandb_projects = ["SL-NAS-common-best"]

    experiments_2_run_ids = OmegaConf.to_container(
            cfg.experiments, resolve=True, throw_on_missing=True)
    dataset2metric = OmegaConf.to_container(
            cfg.dataset2metric, resolve=True, throw_on_missing=True)

    for name, run_info in experiments_2_run_ids.items():
        #if name != "isic2019":
        #    continue
        print(name)
        
        plot(
            dataset2metric=dataset2metric,
            wandb_entity=wandb_entity, 
            wandb_projects=wandb_projects, 
            save_folder_name=save_folder_name,
            save_name=name,
            **run_info
        )

if __name__ == "__main__":
    main()