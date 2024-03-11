from typing import Dict
import wandb
import os
import re
import yaml
import sys
sys.path.append(f"{os.getcwd()}")
from plotting.experiments.d_application.util import *


def get_wandb_low_data_regime_run_ids(
        filters: Dict[str, str],
        project: str,
    ):
    """
    Get the run ids for the low data regime experiments.
    - filters them project them and puts them in a dict by:
        - model name
        - reduction factor
    """
    runs = get_wandb_runs_from_filters(filters, project=project)
    save_run_ids = {}

    for run in runs:
        config = run.config
        name = create_name_comparision_models(config)
        if name not in save_run_ids:
            save_run_ids[name] = {}

        #print("config", config)
        reduction_factor = config["dataset"]["reduction_factor"]

        if reduction_factor not in save_run_ids[name]:
            save_run_ids[name][reduction_factor] = []

        save_run_ids[name][reduction_factor].append(run.id)

    for name, values in save_run_ids.items():
        save_run_ids[name] = dict(sorted(values.items()))
    
    return save_run_ids


        