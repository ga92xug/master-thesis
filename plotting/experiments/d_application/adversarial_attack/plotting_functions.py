import os
import sys
import matplotlib.pyplot as plt
import math

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.plotting_utils import *
from plotting.experiments.plot_acc_flops_params import *
from plotting.experiments.d_application.adversarial_attack.wandb_data import *


def plot_adversarial_attacks(
        save_folder_name: str,
        save_name: str,
        data: Dict,
        plot_type: str = 'combined',  # Either 'individual' or 'combined'
    ):
    
    # order metrics_dict by key so the colors are the same
    data = dict(sorted(data.items(), key=lambda item: item[0]))
    for attack, values in data.items():
        # order values by key so the colors are the same
        data[attack] = dict(sorted(values.items(), key=lambda item: item[0]))

    if plot_type == 'individual':
        for attack, values in data.items():
            fig, ax = plt.subplots()
            
            for model, model_data in values.items():
                color = name2color(model)
                ax.plot(model_data['epsilons'], model_data['robust_accs'], label=model, color=color)
            
            ax.set_title(f'Robust Accuracy vs Epsilon for {attack}')
            ax.set_xlabel('Epsilon')
            ax.set_ylabel('Robust acc [%]')
            ax.legend()
            
            save_plot(
                figure=fig,
                name=save_name + "_" + attack,
                folder_name=save_folder_name,
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
                color = name2color(model)
                ax.plot(model_data['epsilons'], model_data['robust_accs'], label=model, color=color)
                
            ax.set_title(f'{attack}')
            ax.set_xlabel('Epsilon')
            ax.set_ylabel('Robust acc [%]')
            ax.legend()
        
        # Remove any unused subplots
        for idx in range(n_attacks, n_rows * n_cols):
            axs[idx].axis('off')
        
        save_plot(
            figure=fig,
            name=save_name,
            folder_name=save_folder_name,
        )
        plt.close(fig)

    else:
        raise ValueError(f"plot_type {plot_type} not supported.")
    
