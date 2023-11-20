from typing import Dict
import wandb
import os
import re
import yaml
import math
import sys

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.d_application.util import *
from networks.util import flatten_dict
from plotting.experiments.plotting_utils import smooth_data



def get_wandb_efficieny_run_ids(
        filters: Dict[str, str],
    ):
    runs = get_wandb_runs_from_filters(filters)
    save_run_ids = {}

    for run in runs:
        config = run.config
        name = create_name_comparision_models(config)
        if name not in save_run_ids:
            save_run_ids[name] = []


        save_run_ids[name].append(run.id)
    
    return save_run_ids

lenght_datasets = {
    "blood": 11964, # 
    "DeepDRiD_quality": 1200, # 56
    "ISIC_2019": 18237, # epochs 36
}

def get_FLOPs_4_val_values(
        run: wandb.sdk.wandb_run.Run,
        valid_metric: list,
    ):

    GFLOPs_per_image = run.summary["GFLOPs_per_image"] # GFLOPs_per_image
    print("GFLOPs_per_image", GFLOPs_per_image)
    dataset_name = run.config["training"]["dataset"]["name"]

    GFLOPs_per_epoch = GFLOPs_per_image * lenght_datasets[dataset_name]

    # the last epoch is not finished
    max_GFLOPs = run.config["training"]["max_gflops"]

    num_epochs_real = max_GFLOPs / GFLOPs_per_epoch

    num_epochs = run.summary["scheduler"]["epoch"]

    assert num_epochs_real >= num_epochs, f"num_epochs_real {num_epochs_real} < num_epochs {num_epochs}"
    assert len(valid_metric) == math.ceil(num_epochs_real), f"len(valid_metric) {len(valid_metric)} != num_epochs_real {math.ceil(num_epochs_real)}"

    FLOPs_4_val_values = []

    for i in range(len(valid_metric)):
        if num_epochs_real >= 1:
            value = GFLOPs_per_epoch * (i+1)
        else:
            value = FLOPs_4_val_values[-1] + GFLOPs_per_epoch * num_epochs_real

        FLOPs_4_val_values.append(value)
        num_epochs_real -= 1

    assert len(valid_metric) == len(FLOPs_4_val_values), f"len(valid_metric) {len(valid_metric)} != len(FLOPs_4_val_values) {len(FLOPs_4_val_values)}"
    assert FLOPs_4_val_values[-1] == max_GFLOPs, f"FLOPs_4_val_values[-1] {FLOPs_4_val_values[-1]} != max_FLOPs {max_FLOPs}"

    return FLOPs_4_val_values



def get_wandb_data_efficieny_from_ids(
        entity: str,
        project: list,
        labels_run_ids: dict,
        metric_is_weighted: Dict[str, str],
        smoothing_window_size: int = 1,
    ):
    api = wandb.Api()
    data = {}

    if metric_is_weighted:
        valid_metric_name = "valid.acc_weighted"
        test_metric_name = "test.acc_weighted"
    else:
        valid_metric_name = "valid.acc"
        test_metric_name = "test.acc"

    for label, run_ids in labels_run_ids.items():
        print(f"label: {label}")
        data[label] = {}
        for i, run_id in enumerate(run_ids):
            print(f"run_id: {run_id}")
            run = api.run(f"{entity}/{project}/{run_id}")

            valid_metric = run.history(keys=[valid_metric_name]).values[:, 1] * 100
            if smoothing_window_size > 1:
                valid_metric = smooth_data(valid_metric, smoothing_window_size)

            test_metric = run.summary[test_metric_name.split(".")[0]][test_metric_name.split(".")[1]] * 100

            if i == 0:
                FLOPs_4_val_values = get_FLOPs_4_val_values(run, valid_metric)

                data[label] = {
                    "valid_metric": [valid_metric],
                    "test_metric": [test_metric],
                    "FLOPs_4_val_values": FLOPs_4_val_values,
                }

            else:
                data[label]["valid_metric"].append(valid_metric)
                data[label]["test_metric"].append(test_metric)

    print("data", data)
    
    return data, valid_metric_name, test_metric_name