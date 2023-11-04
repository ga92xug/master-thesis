import os
import sys
import numpy as np
from typing import List, Union
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import math

sys.path.append(f"{os.getcwd()}")
from plotting.util import *
from plotting.plot_acc_flops_params import *
from plotting.experiments.d_application._4_adversarial_attack.get_wandb_data import get_wandb_adversarial_attack_data


def plot(
        save_folder_name: str,
        save_name: str,
        data: Dict,
        plot_type: str = 'combined',  # Either 'individual' or 'combined'
    ):
    # order metrics_dict by key so the colors are the same
    data = dict(sorted(data.items(), key=lambda item: item[0]))
    print("data", data.keys())

    if plot_type == 'individual':
        for attack, values in data.items():
            fig, ax = plt.subplots()
            
            for model, model_data in values.items():
                ax.plot(model_data['epsilons'], model_data['robust_accs'], label=model)
            
            ax.set_title(f'Robust Accuracy vs Epsilon for {attack}')
            ax.set_xlabel('Epsilon')
            ax.set_ylabel('Robust Accuracy')
            ax.legend()
            
            save_plot(
                figure=fig,
                name=attack,
                folder_name=save_folder_name + "/" + save_name,
            )
            plt.close(fig)
            
    elif plot_type == 'combined':
        n_attacks = len(data)
        n_cols = 2  # Two attacks per row
        n_rows = math.ceil(n_attacks / n_cols)
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axs = axs.ravel()  # Flatten the array for easier indexing
        
        for idx, (attack, values) in enumerate(data.items()):
            ax = axs[idx]
            
            for model, model_data in values.items():
                ax.plot(model_data['epsilons'], model_data['robust_accs'], label=model)
                
            ax.set_title(f'{attack}')
            ax.set_xlabel('Epsilon')
            ax.set_ylabel('Robust Accuracy')
            ax.legend()
        
        # Remove any unused subplots
        for idx in range(n_attacks, n_rows * n_cols):
            axs[idx].axis('off')
        
        save_plot(
            figure=fig,
            name="combined",
            folder_name=save_folder_name + "/" + save_name,
        )
        plt.close(fig)

    else:
        raise ValueError(f"plot_type {plot_type} not supported.")
    

def main():
    cfg, save_folder_name = plot_init("d_application/adversarial_attacks/", override=True)

    experiments2filters = OmegaConf.to_container(
            cfg.experiments, resolve=True, throw_on_missing=True)

    attack_name_list = [
        "LinfProjectedGradientDescentAttack", 
        "L2ProjectedGradientDescentAttack", 
        "LinfDeepFoolAttack", 
        "L2DeepFoolAttack"
    ]

    for exp_name, values in experiments2filters.items():
        #if exp_name == "Blood":
        #    continue
        filters = values["filters"]
        print("name", exp_name)
        print("filters", filters)
        # get data
        data = get_wandb_adversarial_attack_data(filters, attack_name_list)
             
        plot(
            save_folder_name=save_folder_name,
            save_name=exp_name,
            data=data
        )


if __name__ == "__main__":
    main()    


