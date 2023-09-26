import os
import sys
from typing import List

sys.path.append(f"{os.getcwd()}")

from plot.util import *


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
        wandb_entity: str,
        wandb_projects: str,
        save_folder_name: str,
        save_name: str,
        run_ids: List[str] ,
        labels: List[str], 
        dataset: str,  
        **kwargs,
    ):
    metric = dataset2metric[dataset]

    valid_accs, param_counts, flops = download_data(
        entity=wandb_entity, 
        projects=wandb_projects, 
        run_ids=run_ids,
        metric=metric,
    )

    fig = create_combined_plot(
        valid_accs=valid_accs,
        flops=flops,
        total_params=param_counts,
        long_labels=labels,
        metric=metric,
        **kwargs,
    )

    save_plot(
        figure=fig,
        name=save_name,
        folder_name=save_folder_name,
    )
    

dict_runs_ids = {
    "mnist12k": {
        "run_ids": ["89hdwkuu", "v39l5c9e", "8182cpum"], 
        "labels": ["EQ-NASNET", "EQ-WRN-16-4", "WRN-16-4"],
        "dataset": "mnist12k",
    },

    "mnist_rot": {
        "run_ids": ["379dxvs1", "dvbnwzbz", "stz622pp"], 
        "labels": ["EQ-NASNET", "EQ-WRN-16-4", "WRN-16-4"],
        "dataset": "mnist_rot",
    },

    "cifar10": {
        "run_ids": ["", "", "","tn05kvz2", "9z9q8wvp"], 
        "labels": ["EQ-NASNET", "EQ-WRN-16-4", "NAS-on-CIFAR10", "WRN-16-4", "DenseNet"],
    },

    "galaxy10-eq_nasnet_tests": {
        "run_ids": ["fpnm1nkc", "36jzil3m", "rmuv5wmj", "4784sgy5"], 
        "labels": ["normal", "se0_d0.2_w2", "se0_d0.0_w2", "se0_d0.0_w1"],
        "dataset": "galaxy10",
    },
    "galaxy10-normal": {
        "run_ids": ["36jzil3m", "c3vqa6y2", "iogz6ykb", "dvf3c02p", "jghlotrz"], 
        "labels": ["EQ-NASNET", "EQ-WRN-16-4", "NAS-on-Galaxy10", "DenseNet", "WRN-16-4"],
        "dataset": "galaxy10",
    },

    "isic2019": {
        "run_ids": ["mavho3p2", "axot25dt", "8nf0431h", "yjhex2jy"],
        "labels": ["EQ-NASNET", "EQ-WRN-16-4", "WRN-16-4", "DenseNet"],
        "horizontal_line": {"y": 0.65, "label": "SOTA", "color": "red", "linestyle": "dashed"},
        "dataset": "isic2019",
    },

    
}



def main():
    wandb_entity = "ga92xug"
    wandb_projects = ["SL-NAS-common-best"]
    save_folder_name = "plot/figures/b_NAS/common_best/"

    for name, run_info in dict_runs_ids.items():
        if name != "isic2019":
            continue
        plot(
            wandb_entity=wandb_entity, 
            wandb_projects=wandb_projects, 
            save_folder_name=save_folder_name,
            save_name=name,
            **run_info
        )


if __name__ == "__main__":
    main()