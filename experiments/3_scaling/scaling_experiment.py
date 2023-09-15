import os
import sys
sys.path.append(f"{os.getcwd()}")
from experiments.run_command import run_command

# experiment 1: depth or width scaling laws

# global arguments
global_args = {
    # wandb
    "wandb.project": "SL-Scaling",
    # model
    "model": "eq_wrn",
    "model.restrict": "[none,none]",
    "model.depth": 16,
    "model.widen_factor": 2,
    "training.dataset.resolution": 96,
    # training
    "training": "isic2019-training",
    "training.dataset.batch_size": 32,
    "training.dataset.eval_batch_size": 32,
    "training.accumulate": 1,
}

# baseline
args = {
    "wandb.tags": "[isic2019-baseline]",
}
#run_command(args, global_args, test=False)

# Depth 
for i, depth in enumerate([22, 28, 34, 40, 46]):
    args = {
        "wandb.tags": "[isic2019-depth_scaling]",
        "model.depth": depth,
        "wandb.notes": f"d{i}",
    }
    run_command(args, global_args, test="instantiation")

# Width
for i, width in enumerate([3, 4, 5, 6, 7]):
    args = {
        "wandb.tags": "[isic2019-width_scaling]",
        "model.widen_factor": width,
        "wandb.notes": f"w{i}",
    }
    run_command(args, global_args, test="instantiation")


# resolution scaling
for i, resolution in enumerate([96, 128, 160, 192, 224]):
    args = {
        "wandb.tags": "[isic2019-resolution_scaling]",
        "training.dataset.resolution": resolution,
        "wandb.notes": f"r{i}",
    }
    run_command(args, global_args, test="instantiation")
