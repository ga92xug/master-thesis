import os
import sys
import matplotlib.pyplot as plt
import math

sys.path.append(f"{os.getcwd()}")
from plotting.experiments.plotting_utils import *
from plotting.experiments.plot_acc_flops_params import *
from plotting.experiments.d_application.adversarial_attack.wandb_data import *

def get_title(attack: str):
    title = attack.replace("ProjectedGradientDescentAttack", "PGD")
    title = title.replace("DeepFoolAttack", "DeepFool")
    title = title.replace("L2", "$L_2$")    
    return title

def set_ax_labels(ax, attack: str, idx: int = None):
    ax.set_title(get_title(attack))
    ax.set_xlabel('Epsilon [$\epsilon$]')
    if idx == 0:
        ax.set_ylabel('Robust Accuracy [%]')
    
    ax.grid(True)

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
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model, model_data in values.items():
                color = label2color(model)
                ax.plot(model_data['epsilons'], model_data['robust_accs'], label=model, color=color)
            
            set_ax_labels(ax, attack)
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
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 6 * n_rows), sharey="row")
        axs = axs.ravel()  # Flatten the array for easier indexing
        
        for idx, (attack, values) in enumerate(data.items()):
            ax = axs[idx]
            
            for model, model_data in values.items():
                color = label2color(model)
                ax.plot(model_data['epsilons'], model_data['robust_accs'], label=model, color=color)

            set_ax_labels(ax, attack, idx)
            # get legend handles and labels
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: sorting_key(t[0])))
            # create the legend

            fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.01))
        
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
    
