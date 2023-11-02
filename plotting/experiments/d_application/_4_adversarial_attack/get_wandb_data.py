from typing import Dict, List
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
        attack_name_list: List[str],
    ):
    runs = wandb_connection(filters)
    data = {}

    for run in runs:
        config = run.config
        name = create_name_comparision_models(config)
        if name in data:
            raise ValueError(f"Name {name}, {run.id} already in save_run_ids.")

        data[name] = run2data(run)

    restructed_data = restructuring_data(data, attack_name_list)

    # sort by key
    restructed_data = dict(sorted(restructed_data.items(), key=lambda item: item[0]))

    return restructed_data

def run2data(run):
    # adversarial_attack.L2ProjectedGradientDescentAttack.0.0.robust_acc
    adversarial_attacks = run.summary["adversarial_attack"]
    #print(run.summary)
    history = run.history()
    #print("history", history)
    # all columns that have the word adversarial_attack
    adversarial_attack_columns = [column for column in history.columns if "adversarial_attack" in column]
    adversarial_attack_columns = sorted(adversarial_attack_columns)
    #print("adversarial_attack_columns", adversarial_attack_columns)

    results = {}

    for column in adversarial_attack_columns:
        if len(column.split(".")) != 5:
            continue

        attack_column = run.history(keys=[column])
        last_value = attack_column.values[-1, 1]    

        # get the attack name
        attack_name = column.split(".")[1] 
        epsilon = column.split(".")[2:4]
        epsilon = ".".join(epsilon)
        metric = column.split(".")[-1]

        if attack_name not in results:
            results[attack_name] = {
                "robust_accs": [],
                "count_advs": [],
                "epsilons": [],
            }
        if metric == "robust_acc":
            results[attack_name]["robust_accs"].append(last_value)
            results[attack_name]["epsilons"].append(epsilon)
        elif metric == "count_adv":
            results[attack_name]["count_advs"].append(last_value)

    return results
    #print("history", history)
    """
    now the values are not in summary anymore

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
    """

    

    
def restructuring_data(data: Dict, attack_name_list: List):
    """
    """

    restructured_data = {}
    # get attacks from first model
    #attack_name_list = list(data.values())[0].keys()

    # all models have the same attacks
    for attack in attack_name_list:
        restructured_data[attack] = {}

    for model, values in data.items():
        attack_current = values.keys()
        #assert attack_current == attacks, f"All models should have the same attacks. {attack_current} != {attacks}"

        for attack in attack_name_list:
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


        