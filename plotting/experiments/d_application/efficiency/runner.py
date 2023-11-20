import os
import sys
import numpy as np
from typing import List, Union
from omegaconf import OmegaConf

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.plotting_utils import *
from plotting.experiments.plot_acc_flops_params import *
from plotting.experiments.d_application.efficiency.wandb_data import get_wandb_data_efficieny_from_ids, get_wandb_efficieny_run_ids
from plotting.experiments.d_application.efficiency.plotting_functions import plot_model_efficiency_with_test_acc_histogram


def one_efficiency_plot(
        metric_is_weighted: Dict[str, str],
        wandb_entity: str,
        wandb_project: str,
        save_folder_name: str,
        save_name: str,
        fig_size: Tuple[int, int],
        model2label_run_ids_dict: Dict[str, Union[List[str], str]],
        **kwargs,
    ):

    data, valid_metric_name, test_metric_name = get_wandb_data_efficieny_from_ids(
        entity=wandb_entity, 
        project=wandb_project, 
        labels_run_ids=model2label_run_ids_dict,
        metric_is_weighted=metric_is_weighted,
        smoothing_window_size=1,
    )


    #fig_size = get_fig_size((8,4))
    fig = plot_model_efficiency_with_test_acc_histogram(
        model_data=data,
        valid_metric_name=valid_metric_name,
        test_metric_name=test_metric_name,
    )

    save_plot(
        figure=fig,
        name=save_name,
        folder_name=save_folder_name,
    )
    

def main():
    cfg, save_folder_name = plot_init("d_application/efficiency", override=True)
    dataset2metric = cfg.dataset2metric

    experiments2filters = OmegaConf.to_container(
            cfg.experiments, resolve=True, throw_on_missing=True)
    #print("experiments2filters", experiments2filters)

    for exp_name, values in experiments2filters.items():
        if exp_name != "DeepDRiD":
            continue

        metric_is_weighted = True if "weighted" in dataset2metric[exp_name] else False
        
        if isinstance(values, list):
            # This is not a experiment
            continue

        filters = values["filters"]
        print("name", exp_name)
        print("filters", filters)
        model2run_ids = get_wandb_efficieny_run_ids(filters)
             
        one_efficiency_plot(
            metric_is_weighted=metric_is_weighted,
            wandb_entity=cfg.wandb_entity, 
            wandb_project="SL-Application", 
            save_folder_name=save_folder_name,
            save_name=exp_name,
            fig_size=cfg.figsize,
            model2label_run_ids_dict=model2run_ids
        )


if __name__ == "__main__":
    main()