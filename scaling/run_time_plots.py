import torch
from tqdm import tqdm
import numpy as np
np.set_printoptions(precision=4, linewidth=10000, suppress=True)

from omegaconf import OmegaConf
from hydra import compose, initialize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import sys
import re
sys.path.append('..') # add parent directory
sys.path.append('../scaling-laws-ecnn') # add parent directory
from experiment.network_instantiation import main
from scaling.util import plot_model_data, binary_search_over_model_scaling
from experiment.run_files.run_command import run_command

def extract_info(output):
    # Define the regex patterns to extract the information
    param_count_pattern = r"Total params: (\d+)"
    model_building_time_pattern = r"Model building time: ([\d.]+)"
    train_time_pattern = r"Train time elapsed: ([\d.]+)"
    flops_pattern = r"Flops: ([\d.]+)"
    # Extract the information using regex
    param_count_match = re.search(param_count_pattern, output)
    model_building_time_match = re.search(model_building_time_pattern, output)
    train_time_match = re.search(train_time_pattern, output)
    flops_match = re.search(flops_pattern, output)

    # Extracted values
    param_count = int(param_count_match.group(1)) if param_count_match else None
    model_building_time = float(model_building_time_match.group(1)) if model_building_time_match else None
    train_time = float(train_time_match.group(1)) if train_time_match else None
    flops = float(flops_match.group(1)) if flops_match else None

    return param_count, model_building_time, train_time, flops


def main():
    global_args = [
        "model=wrn", 
        #"model.restrict=[halved,invariant]",
        #"model.kernel_layout=[3,3]", 
        #"model.padding=1",
        "wandb.mode=disabled",
        "training.epochs=1",
        #"model.rotation=8",
        #"model.fix_params_mode=heuristic",
        "training.steps_per_epoch=5000",
    ]

    image_size = 32

    stats = {}
    mode_stats = []
    for depth in range(6, 48, 6):
        args=[f"model.depth={depth+4}",
            f"model.widen_factor={1}",
            f"dataset.resolution={image_size}"]
        output = run_command(args, global_args, test="instantiation")
        param_count, model_building_time, train_time, gflops = extract_info(output)
        #run_command(args, global_args, test="instantiation")
        mode_stats.append([depth, param_count, model_building_time, train_time, gflops])

    stats[f"wrn_X_1"] = np.array(mode_stats)

    # width
    mode_stats = []
    for width in range(10, 40, 5):
        width = width / 10.0
        args=[f"model.depth={10}",
                    f"model.widen_factor={width}",
                    f"dataset.resolution={image_size}"]
        output = run_command(args, global_args, test="instantiation")
        param_count, model_building_time, train_time, gflops = extract_info(output)
        #run_command(args, global_args, test="instantiation")
        mode_stats.append([width, param_count, model_building_time, train_time, gflops])
        
    stats[f"wrn_10_X"] = np.array(mode_stats)
    plot_model_data(stats, vs_param="depth or width")


if __name__ == "__main__":
    main()