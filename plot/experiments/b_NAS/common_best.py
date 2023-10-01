import os
import sys
from typing import List, Union
from omegaconf import OmegaConf

sys.path.append(f"{os.getcwd()}")
from plot.util import *
from plot.plot_functions import *


dict_runs_ids = {
    "mnist12k": {
        "labels_run_ids": {
            "EQ-NASNET": ["d9fwlmg6"],
            "EQ-WRN-16-4": ["v39l5c9e"],
            "WRN-16-4": ["8182cpum"],
        },
        "dataset": "mnist12k",
    },

    "mnist_rot": {
        "labels_run_ids": {
            "EQ-NASNET": ["emm8y2em"],
            "EQ-WRN-16-4": ["dvbnwzbz"],
            "NAS-on-MNIST_rot": ["3tfzbuct"],
            "WRN-16-4": ["stz622pp"],
        },
        "dataset": "mnist_rot",
    },

    "cifar10": {
        "labels_run_ids": {
            "EQ-NASNET": ["oj1twq4i"],
            "EQ-WRN-16-4": ["lzpxj79q"],
            "NAS-on-CIFAR10": ["o9tte2ci"],
            "WRN-16-4": ["tn05kvz2"],
            "DenseNet": ["9z9q8wvp"],
        },
        "dataset": "cifar10",
    },

    "galaxy10-eq_nasnet_tests": {
        "labels_run_ids": {
            "normal": ["fpnm1nkc"],
            "se0_d0.2_w2": ["36jzil3m"],
            "se0_d0.0_w2": ["rmuv5wmj"],
            "se0_d0.0_w1": ["4784sgy5"],
        },
        "dataset": "galaxy10",
    },
    "galaxy10-normal": {
        "labels_run_ids": {
            "EQ-NASNET": ["36jzil3m"],
            "EQ-WRN-16-4": ["c3vqa6y2"],
            "NAS-on-Galaxy10": ["iogz6ykb"],
            "DenseNet": ["dvf3c02p"],
            "WRN-16-4": ["jghlotrz"],
        },
        "dataset": "galaxy10",
    },

    "isic2019": {
        "labels_run_ids": {
            "EQ-NASNET": ["sn2lrp4a", "si8ubh8x", "n2exzy14"],
            "EQ-WRN-16-4": ["f2fgkmto"],
            "WRN-16-4": ["n7i1jtw6"],
            "DenseNet": ["d6mr3c2r"],
        },
        "horizontal_line": {"y": 0.65, "label": "SOTA", "color": "red", "linestyle": "dashed"},
        "dataset": "isic2019",
    },
}

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

    downloaded_data = download_data(
        entity=wandb_entity, 
        projects=wandb_projects, 
        labels_run_ids=labels_run_ids,
        metric=metric,
    )

    fig_size = get_fig_size((8,4))
    print("fig_size", fig_size)

    fig = create_combined_plot(
        downloaded_data=downloaded_data,
        metric=metric,
        fig_size=fig_size,
        **kwargs,
    )

    save_plot(
        figure=fig,
        name=save_name,
        folder_name=save_folder_name,
    )
    

def main():
    cfg, save_folder_name = plot_init("b_NAS/common_best/", override=True)
    wandb_entity = cfg.wandb.entity
    wandb_projects = cfg.wandb.projects

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