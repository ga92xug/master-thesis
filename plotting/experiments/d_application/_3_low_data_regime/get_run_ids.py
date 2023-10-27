from typing import Dict
import wandb
import os
import re
import yaml




def create_name_run(config):
    name = config["model"]["_target_"].split(".")[-1]
    pretrained = config["model"].get("pretrained", False)
    if pretrained:
        name += "_pre"

    return name

def get_runs():
    api = wandb.Api()
    filters = {
        "tags": "test_performance_HPO",
        "config.training.earlystop.monitor": "valid.loss",
    }
    runs = api.runs(path=f"ga92xug/SL-Application", filters=filters)

    print("Number of runs:", len(runs))
    return runs

def save_2_yaml(save_run_ids: Dict[str, Dict[float, list]], file_name: str):
    # Create a YAML file in the current directory
    with open(file_name, "w") as file:
        yaml.dump(save_run_ids, file)
    print("YAML file saved.")



def main():
    runs = get_runs()
    save_run_ids = {}

    for run in runs:
        config = run.config
        name = create_name_run(config)
        if name not in save_run_ids:
            save_run_ids[name] = {}

        reduction_factor = config["training"]["dataset"]["reduction_factor"]


        if reduction_factor not in save_run_ids[name]:
            save_run_ids[name][reduction_factor] = []

        save_run_ids[name][reduction_factor].append(run.id)


    #print(save_run_ids)
    save_2_yaml(save_run_ids, "run_ids.yaml")

if __name__ == "__main__":
    main()


        