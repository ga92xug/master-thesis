import os
import sys
import numpy as np
from typing import List, Union
from omegaconf import OmegaConf
from sympy import O
sys.path.append(f"{os.getcwd()}")
from plotting.util import *
from plotting.plot_functions import *
from plotting.experiments.d_application._3_low_data_regime.plot import plot_metric_vs_reduction


def restructure_data(data: Dict, metric: str):
    restructured_data = {}
    for label, values in data.items():
        factor = label # .split("=")[-1]

        metric_list = []
        for run_id, run_data in values.items():
            metric_list.append(run_data[metric][0])

        if len(metric_list) > 0:
            restructured_data[factor] = np.array(metric_list)

    #print("restructured_data", restructured_data)
    return restructured_data


def plot(
        metric: Dict[str, str],
        wandb_entity: str,
        wandb_projects: str,
        save_folder_name: str,
        save_name: str,
        model2label_run_ids_dict: Dict[str, Union[List[str], str]],
        **kwargs,
    ):
    model2data = {}
    for model_name, value in model2label_run_ids_dict.items():
        labels_run_ids = value # ["labels_run_ids"]
        
        downloaded_data = download_data(
            entity=wandb_entity, 
            projects=wandb_projects, 
            labels_run_ids=labels_run_ids,
            metric=metric,
            name_param_count="param_count"
        )
        data = restructure_data(downloaded_data, metric)
        model2data[model_name] = data


    fig_size = get_fig_size((8,4))
    fig = plot_metric_vs_reduction(
        metrics_dict=model2data,
        metric_name=metric,
        fig_size=fig_size,
        errorbar=False,
        **kwargs,
    )

    save_plot(
        figure=fig,
        name=save_name,
        folder_name=save_folder_name,
    )
    

def main():
    cfg, save_folder_name = plot_init("d_application/low_data_regime/", override=True)
    wandb_entity = cfg.wandb.entity
    wandb_projects = ["SL-Application"]
    metric = "test.acc"

    experiments_2_run_ids = OmegaConf.to_container(
            cfg.experiments, resolve=True, throw_on_missing=True)["experiments"]
    
    for name, model2label_run_ids_dict in experiments_2_run_ids.items():
        print("name", name)
        
        plot(
            metric=metric,
            wandb_entity=wandb_entity, 
            wandb_projects=wandb_projects, 
            save_folder_name=save_folder_name,
            save_name=name,
            model2label_run_ids_dict=model2label_run_ids_dict
        )


if __name__ == "__main__":
    main()