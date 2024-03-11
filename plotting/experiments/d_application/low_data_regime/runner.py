import os
import sys
import numpy as np
from typing import List, Union
from omegaconf import OmegaConf

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.plotting_utils import *
from plotting.experiments.plot_acc_flops_params import *
from plotting.experiments.wandb_utils import get_wandb_data_multiple_runs
from plotting.experiments.d_application.low_data_regime.wandb_data_2 import get_wandb_low_data_regime_run_ids
from plotting.experiments.d_application.low_data_regime.plotting_functions import plot_low_data_regime


def restructure_data(
        data: Dict, 
        metric: str
    ):
    restructured_data = {}
    for label, values in data.items():
        factor = label

        metric_list = []
        for run_id, run_data in values.items():
            metric_list.append(run_data[metric][0])

        if len(metric_list) > 0:
            restructured_data[factor] = np.array(metric_list)

    return restructured_data


def one_low_data_regime_plot(
        metric: str,
        wandb_entity: str,
        wandb_projects: str,
        save_folder_name: str,
        save_name: str,
        fig_size: Tuple[int, int],
        model2label_run_ids_dict: Dict[str, Union[List[str], str]],
        **kwargs,
    ):
    model2data = {}
    for model_name, value in model2label_run_ids_dict.items():
        print("Model name:", model_name)
        labels_run_ids = value # ["labels_run_ids"]
        print("labels_run_ids", labels_run_ids)

        data = get_wandb_data_multiple_runs(
            entity=wandb_entity, 
            projects=wandb_projects, 
            labels_run_ids=labels_run_ids,
            metric=metric,
            name_param_count="param_count"
        )
        data = restructure_data(data, metric)
        model2data[model_name] = data


    #fig_size = get_fig_size((8,4))
    fig = plot_low_data_regime(
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
    cfg, save_folder_name = plot_init("d_application/low_data_regime", override=True)
    metric = "test/acc"

    experiments2filters = OmegaConf.to_container(
            cfg.experiments, resolve=True, throw_on_missing=True)

    for exp_name, values in experiments2filters.items():
        if isinstance(values, list) or isinstance(values, str):
            # This is not a experiment
            continue

        filters = values["filters"]
        print("name", exp_name)
        print("filters", filters)
        # Get the run ids for the low data regime experiments.
        model2run_ids = get_wandb_low_data_regime_run_ids(filters, project=cfg.experiments.wandb_projects)
        
        # with the run ids plot the data
        one_low_data_regime_plot(
            metric=metric,
            wandb_entity=cfg.wandb_entity, 
            wandb_projects=cfg.experiments.wandb_projects, 
            save_folder_name=save_folder_name,
            save_name=exp_name,
            fig_size=cfg.figsize,
            model2label_run_ids_dict=model2run_ids
        )


if __name__ == "__main__":
    main()