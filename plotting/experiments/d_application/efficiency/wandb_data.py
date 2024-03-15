from typing import Dict
import wandb
import os
import re
import yaml
import math
import sys
import numpy as np

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
    "blood": 9571, # 
    "DeepDRiD_quality": 1200, # 56
    "ISIC_2019": 18237, # epochs 36
}

def get_FLOPs_4_val_values_old(
        run: wandb.sdk.wandb_run.Run,
        valid_metric: list,
    ):

    GFLOPs_per_image = run.summary["GFLOPs_per_image"] # GFLOPs_per_image
    print("GFLOPs_per_image", GFLOPs_per_image)
    dataset_name = run.config["training"]["dataset"]["name"]
    batch_size = run.config["training"]["dataset"]["batch_size"]
    eval_frequency = run.config["training"]["eval_frequency"]
    train_dataset_len = lenght_datasets[dataset_name]
    train_n_batches_len = math.ceil(train_dataset_len / batch_size)
    GFLOPs_per_epoch = GFLOPs_per_image * lenght_datasets[dataset_name]
    print("GFLOPs_per_epoch", GFLOPs_per_epoch)

    if eval_frequency < 1 and eval_frequency > 0:
        eval_points = [int(train_n_batches_len * eval_frequency * i) for i in range(1, int(1/eval_frequency))]
        print("eval_points", eval_points)
        GFLOPs_per_batch = GFLOPs_per_image * batch_size

    # the last epoch is not finished
    max_GFLOPs = run.config["training"]["max_gflops"]

    num_epochs_real = max_GFLOPs / GFLOPs_per_epoch

    num_epochs = run.summary["scheduler"]["epoch"]
    print("finished epochs", num_epochs)

    assert num_epochs_real >= num_epochs, f"num_epochs_real {num_epochs_real} < num_epochs {num_epochs}"
    if not (eval_frequency < 1 and eval_frequency > 0):
        assert len(valid_metric) == math.ceil(num_epochs_real), f"len(valid_metric) {len(valid_metric)} != num_epochs_real {math.ceil(num_epochs_real)}"
    else:
        print("len(valid_metric)", len(valid_metric))
        print("len(eval_points) + 1", len(eval_points) + 1)

    FLOPs_4_val_values = []
    
    current_frequency = eval_frequency
    current_point = 0
    for i in range(len(valid_metric)):
        if i == len(valid_metric) - 1:
            value = max_GFLOPs - FLOPs_4_val_values[-1]
            print("last epoch")

        else:

            if eval_frequency < 1 and eval_frequency > 0:
                if current_frequency + eval_frequency < 1:
                    if current_point > 0:
                        num_batches = eval_points[current_point] - eval_points[current_point - 1]                
                    else:
                        num_batches = eval_points[current_point]
                    value = GFLOPs_per_batch * num_batches
                    print("once per eval point", current_frequency)
                    current_frequency += eval_frequency
                    current_point += 1
                    
                else:
                    # epoch end
                    current_frequency = eval_frequency
                    current_point = 0
                
                    value = GFLOPs_per_epoch - GFLOPs_per_batch * eval_points[-1]
                    print("once per epoch")


            else:
                value = GFLOPs_per_epoch
                num_epochs_real -= 1

        if len(FLOPs_4_val_values) > 0:
            value += FLOPs_4_val_values[-1]

        print("FLOPs", value)
        FLOPs_4_val_values.append(value)

    assert len(valid_metric) == len(FLOPs_4_val_values), f"len(valid_metric) {len(valid_metric)} != len(FLOPs_4_val_values) {len(FLOPs_4_val_values)}"

    return FLOPs_4_val_values


def get_FLOPs_4_val_values(
        run: wandb.sdk.wandb_run.Run,
        valid_metric: list,
    ):
    GFLOPs_per_image = run.summary["GFLOPs_per_image"] # GFLOPs_per_image
    dataset_name = run.config["training"]["dataset"]["name"]
    batch_size = run.config["training"]["dataset"]["batch_size"]
    eval_frequency = run.config["training"]["eval_frequency"]
    train_dataset_len = lenght_datasets[dataset_name]
    train_n_batches_len = math.ceil(train_dataset_len / batch_size)
    
    
    max_GFLOPs = run.config["training"]["max_gflops"]

    if eval_frequency < 1 and eval_frequency > 0:
        eval_points = [int(train_n_batches_len * eval_frequency * i) for i in range(1, int(1/eval_frequency))]
        GFLOPs_per_batch = GFLOPs_per_image * batch_size

    if not (eval_frequency < 1 and eval_frequency > 0):
        GFLOPs_per_epoch = GFLOPs_per_image * lenght_datasets[dataset_name]
        starting_point = GFLOPs_per_epoch
    else:
        starting_point = GFLOPs_per_batch * eval_points[0]

    FLOPs_4_val_values = np.linspace(starting_point, max_GFLOPs, len(valid_metric))

    return FLOPs_4_val_values

    

def get_wandb_data_efficieny_from_ids(
        entity: str,
        project: list,
        labels_run_ids: dict,
        metric_is_weighted: Dict[str, str],
        smoothing_window_size: int = 3,
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
            #print(f"run_id: {run_id}")
            run = api.run(f"{entity}/{project}/{run_id}")

            valid_metric = run.history(keys=[valid_metric_name]).values[:, 1] * 100
            if smoothing_window_size > 1:
                valid_metric = smooth_data(valid_metric, smoothing_window_size, keep_size=True)

            test_metric = run.summary[test_metric_name.split(".")[0]][test_metric_name.split(".")[1]] * 100

            if i == 0:
                FLOPs_4_val_values = get_FLOPs_4_val_values(run, valid_metric)
                epochs = run.summary["scheduler"]["epoch"]
                if epochs == 0:
                    epochs = 1
                
                data[label] = {
                    "valid_metric": [valid_metric],
                    "test_metric": [test_metric],
                    "FLOPs_4_val_values": FLOPs_4_val_values,
                    "epochs": epochs,
                }

            else:
                data[label]["valid_metric"].append(valid_metric)
                data[label]["test_metric"].append(test_metric)

    #print("data", data)
    
    return data, valid_metric_name, test_metric_name