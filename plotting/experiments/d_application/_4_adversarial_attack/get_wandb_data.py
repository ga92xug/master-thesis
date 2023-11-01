from typing import Dict
import wandb
import os
import re
import sys
import yaml
sys.path.append(f"{os.getcwd()}")
from plotting.experiments.d_application.util import create_name_comparision_models

def wandb_connection(filters: Dict[str, str]):
    api = wandb.Api()
    filters["state"] = "finished"
    runs = api.runs(path=f"ga92xug/SL-Application", filters=filters)

    print("Number of runs:", len(runs))
    return runs

def get_wandb_adversarial_attack_data(
        filters: Dict[str, str],
    ):
    runs = wandb_connection(filters)
    data = {}

    for run in runs:
        config = run.config
        name = create_name_comparision_models(config)
        if name in data:
            raise ValueError(f"Name {name}, {run.id} already in save_run_ids.")

        data[name] = run2data(run)

    restructed_data = restructuring_data(data)

    # sort by key
    restructed_data = dict(sorted(restructed_data.items(), key=lambda item: item[0]))

    return restructed_data

def run2data(run):
    # adversarial_attack.L2ProjectedGradientDescentAttack.0.0.robust_acc
    adversarial_attacks = run.summary["adversarial_attack"]

    results = {}

    for k, v in adversarial_attacks.items():
        if isinstance(v, int):
            # all the attacks have at least one epsilon
            continue

        results[k] = {
            "robust_accs": [],
            "count_advs": [],
            "epsilons": [],
        }

        # Extract epsilons and epsilon_dict into a list of tuples
        epsilon_data = [(float(epsilon), epsilon_dict) for epsilon, epsilon_dict in v.items()]

        # Sort the list of tuples based on epsilon values
        epsilon_data.sort(key=lambda x: x[0])

        for epsilon, epsilon_dict in epsilon_data:
            results[k]["robust_accs"].append(epsilon_dict["robust_acc"])
            results[k]["count_advs"].append(epsilon_dict["count_adv"])
            results[k]["epsilons"].append(epsilon)

    return results

    
def restructuring_data(data: Dict):
    """
    """

    restructured_data = {}
    # get attacks from first model
    attacks = list(data.values())[0].keys()

    # all models have the same attacks
    for attack in attacks:
        restructured_data[attack] = {}

    for model, values in data.items():
        attack_current = values.keys()
        assert attack_current == attacks, f"All models should have the same attacks. {attack_current} != {attacks}"

        for attack in attacks:
            restructured_data[attack][model] = {}
            restructured_data[attack][model]["epsilons"] = values[attack]["epsilons"]
            restructured_data[attack][model]["robust_accs"] = values[attack]["robust_accs"]
            restructured_data[attack][model]["count_advs"] = values[attack]["count_advs"]

        
    return restructured_data


if __name__ == "__main__":
    filters = {
        "config.training.adversarial_attack": "Foolbox",
        "config.training.epochs": 100, # full training
    }

    result = get_wandb_adversarial_attack_data(filters)
    print(result)


        