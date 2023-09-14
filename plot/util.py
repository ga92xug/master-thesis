import os
import sys
import matplotlib.pyplot as plt
from numpy import save
import pandas as pd
import wandb
from matplotlib import pyplot as plt, ticker
sys.path.append('../scaling-laws-ecnn') # add parent directory

from training.speed_test import main

title_size = 12
label_size = 10

def plot_model_data(model_data, vs_param):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
    fig.suptitle(f'Scaling {vs_param}', fontsize=14, fontweight='bold')
    
    for model_name, data in model_data.items():
        vs = data[:, 0]
        param_count = data[:, 1]
        model_building_time = data[:, 2]
        train_time = data[:, 3]
        gflops = data[:, 4]
        
        ax[0].plot(vs, param_count, label=model_name)
        ax[1].plot(vs, model_building_time, label=model_name)
        ax[2].plot(vs, train_time, label=model_name)
        ax[3].plot(vs, gflops, label=model_name)
    
    ax[0].set_xlabel(f'{vs_param}')
    ax[0].set_ylabel('Parameter Count')
    ax[0].set_title(f'Parameter Count')
    ax[0].legend()

    ax[1].set_xlabel(f'{vs_param}')
    ax[1].set_ylabel('Model Building Time')
    ax[1].set_title(f'Model Building Time')
    ax[1].legend()

    ax[2].set_xlabel(f'{vs_param}')
    ax[2].set_ylabel('Train Time')
    ax[2].set_title(f'Train Time')
    ax[2].legend()

    ax[3].set_xlabel(f'{vs_param}')
    ax[3].set_ylabel('GFLOPs')
    ax[3].set_title(f'GFLOPs')
    ax[3].legend()

    plt.tight_layout()
    plt.savefig(f'scaling/figures/{model_name.split("_")[0]}_{vs_param}_scaling.png')


def download_data(entity, project, run_ids):    
    valid_accs = []
    total_params = []
    train_times = []
    gflops = []
    
    # Download validation accuracy, total parameters, and train times for each run
    for run_id in run_ids:
        run = wandb.Api().run(f"{entity}/{project}/{run_id}")
        valid_accs.append(run.history(keys=['valid.acc']).values[:, 1])
        
        # Download total parameters
        total_params.append(run.history(keys=['total_parameters']).values[:, 1][0])
        
        # Download train times
        #train_times.append(run.summary.get('train_time', 0))  # Replace 'train_time' with the actual key name

        # GFLOPs
        try:
            gflops.append(run.history(keys=['GFLOPs']).values[:, 1][0])
        except:
            pass

    return valid_accs, total_params, gflops


def smooth_data(data, window=5):
    df = pd.DataFrame(data)
    smoothed_data = df.rolling(window=window, axis=1, min_periods=1).mean().values
    return smoothed_data

def save_plot(figure, location, name, folder_name='figures/first_experiments'):
    name = name.replace(' ', '_').lower()

    if not os.path.exists(f'{folder_name}'):
        os.makedirs(f'{folder_name}')

    figure.savefig(f'{folder_name}/{location}_{name}.png', dpi=300, bbox_inches = "tight")


def plot_validation_accuracy(valid_accs, lables, title):
    valid_accs = smooth_data(valid_accs)
    figure = plt.figure(figsize=(4, 3))
    for i, acc in enumerate(valid_accs):
        plt.plot(acc, label=lables[i], linewidth=3)
    plt.xlabel('Epoch', fontsize=label_size)
    plt.ylabel('Validation Accuracy', fontsize=label_size)
    plt.ylim(0.6, 1.0)
    #plt.title(title, fontsize=title_size, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])

    save_plot(figure, "valid_acc", title)
    plt.show()


def plot_total_parameters(total_params, labels, title):
    plt.figure(figsize=(2, 3))
    colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    plt.bar(labels, total_params, color=colors)
    #plt.xlabel('Run ID')
    #plt.title('# Parameters', fontsize=title_size, fontweight='bold')
    #plt.title(title)
    #plt.grid(axis='y')
    plt.ylabel('# Parameters')
    save_plot(plt, "params", title)
    plt.show()

def plot_train_times(train_times, labels, title):
    plt.figure(figsize=(2, 3))
    colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    plt.bar(labels, train_times, color=colors)
    #plt.xlabel('Run ID')
    #plt.title('Train Time (seconds)', fontsize=title_size, fontweight='bold')
    #plt.title(title)
    #plt.grid(axis='y')
    name = title.replace(' ', '_').lower()
    save_plot(plt, "train_time", title)
    plt.show()

def plot_flops(gflops, labels, title):
    plt.figure(figsize=(2, 3))
    colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    plt.bar(labels, gflops, color=colors)

    # Set the y-axis tick formatter
    #formatter = ticker.FuncFormatter(lambda x, pos: f'{x / 1e2:.1f}')
    #plt.gca().yaxis.set_major_formatter(formatter)
    plt.ylabel('FLOPs')
    #plt.xlabel('Run ID')
    #plt.title('GFLOPs', fontsize=title_size, fontweight='bold')
    #plt.title(title)
    #plt.grid(axis='y')
    name = title.replace(' ', '_').lower()
    save_plot(plt, "flops", title)
    plt.show()




def binary_search_over_model_scaling(
        search_param, scale_param, scaling_factor, initial_range, constraint_func,
        overrides, global_overrides, max_iterations=50, tolerance=0.01):
    """
    does not work for now

    Perform a binary search to find the best scaling parameter for a given model configuration.

    Args:
        search_param (dict): A dictionary containing the parameter to search for (key) and its initial value (value).
        scale_param (dict): A dictionary containing the parameter to scale for (key) and its initial value (value).
        scaling_factor (float): The desired scaling factor for the scale_param.
        initial_range (tuple): A tuple containing the initial lower and upper bounds for the search_param.
        constraint_func (function): A function to apply model-specific constraints to the search_param.
        overrides (dict): A dictionary containing model configuration parameters that are not part of the search.
        global_overrides (dict): A dictionary containing global configuration parameters for the model.
        max_iterations (int, optional): The maximum number of iterations for the binary search. Default is 100.
        tolerance (float, optional): The acceptable error threshold for convergence. Default is 0.01.

    Returns:
        mid: The best scaling parameter found for the search_param.

    Raises:
        ValueError: If the search_param or scale_param is not valid or supported.
    """
    assert False, "not supported for now"

    scale_indices = {
        "param_count": 0,
        "model_building_time": 1,
        "train_time": 2
    }
    
    search_key = list(search_param.keys())[0]
    scale_key = list(scale_param.keys())[0]
    scale_index = scale_indices[scale_key]

    initial_value = search_param[search_key]
    local_overrides = {**overrides, **search_param}
    total_overrides = global_overrides.copy()
    total_overrides.update(local_overrides)

    base_stats = run(overrides=total_overrides, verbose=0)

    lower_bound, upper_bound = initial_range

    is_integer = isinstance(initial_value, int)

    iteration = 0
    while iteration < max_iterations:
        mid = (lower_bound + upper_bound) / 2
        if is_integer:
            mid = int(mid)

        mid = constraint_func(mid)
        local_overrides[search_key] = mid
        total_overrides.update(local_overrides)

        try:
            stats = run(overrides=total_overrides, verbose=0)
        except:
            upper_bound = mid * 0.9
            continue

        if abs(stats[scale_index] - base_stats[scale_index] * scaling_factor) <= tolerance:
            break
        elif stats[scale_index] > base_stats[scale_index] * scaling_factor:
            upper_bound = mid * 0.9
        else:
            lower_bound = mid * 1.1

        iteration += 1

    print(f"Best {search_key} found: {mid}")
    return mid

