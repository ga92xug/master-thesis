import os
import sys
from omegaconf import OmegaConf
sys.path.append(f"{os.getcwd()}")
from plotting.experiments.plotting_utils import *
from plotting.experiments.plot_acc_flops_params import *
from plotting.experiments.d_application.adversarial_attack.wandb_data import *
from plotting.experiments.d_application.adversarial_attack.plotting_functions import *


def main():
    cfg, save_folder_name = plot_init("d_application/adversarial_attacks", override=True)
    experiments2filters = OmegaConf.to_container(
            cfg.experiments, resolve=True, throw_on_missing=True)

    attack_name_list = [
        #"LinfProjectedGradientDescentAttack", 
        "L2ProjectedGradientDescentAttack", 
        #"LinfDeepFoolAttack", 
        "L2DeepFoolAttack",
    ]

    for exp_name, values in experiments2filters.items():
        #if exp_name == "Blood":
        #    continue
        if not isinstance(values, dict):
            # This is not a experiment
            continue

        filters = values["filters"]
        print("name", exp_name)
        print("filters", filters)
        # get data
        data = get_wandb_adversarial_attack_data(filters, attack_name_list)
             
        plot_adversarial_attacks(
            save_folder_name=save_folder_name,
            save_name=exp_name,
            data=data,
            plot_type='individual',
        )


if __name__ == "__main__":
    main()    


