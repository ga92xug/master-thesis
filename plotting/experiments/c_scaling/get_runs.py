from typing import Dict, Any
import wandb
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(f"{os.getcwd()}")

from plotting.experiments.c_scaling.plot_scaling import *
from plotting.util import plot_init, download_run
from plotting.plot_functions import get_metric_from_downloaded_data


def get_wandbdata_with_filters(
        wandb_entity: str,
        wandb_projects: str,
        filters: Dict[str, Any],
    ):
    api = wandb.Api()
    runs = api.runs(path=f"{wandb_entity}/{wandb_projects}", filters=filters)

    results = {}
    for run in runs:
        results[run.id] = download_run(run=run, metric="valid.acc_weighted")


    return results


def main():
    cfg, save_folder_name = plot_init("c_scaling/individual/")
    wandb_entity = cfg.wandb.entity
    wandb_projects = "SL-Scaling"


    filters = {
        "config.training.epochs": 120,
        #"tags": ["baseline_isic2019", "width_scaling"],
        "tags": "baseline_isic2019",
    }
        
    results_for_filter = get_wandbdata_with_filters(
        wandb_entity=wandb_entity,
        wandb_projects=wandb_projects,
        filters=filters,
    )

if __name__ == "__main__":
    main()