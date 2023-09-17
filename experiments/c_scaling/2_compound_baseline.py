import os
import re
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




# Depth 
# 28: 0992 GFLOPs
# compound scaling every 12 layers ca. +500 GFLOPs
for i, depth in enumerate([28, 40]):
    args = {
        "wandb.tags": "[isic2019-depth_scaling]",
        "model.depth": depth,
        "wandb.notes": f"d{i}",
    }
    #run_command(args, global_args, test="instantiation")

# Width
# 2.5:  722 GFLOPs
# 3:   1032 GFLOPs
# 3.5: 1396 GFLOPs
# 4:   1816 GFLOPs
# 4.5: 2291 GFLOPs
for i, width in enumerate([3, 5]):
    args = {
        "wandb.tags": "[isic2019-width_scaling]",
        "model.widen_factor": width,
        "wandb.notes": f"w{i}",
    }
    #run_command(args, global_args, test="instantiation")


# resolution scaling
# 96:   468 GFLOPs 
# 128:  831 GFLOPs
# 160: 1297 GFLOPs
# 192: 1868 GFLOPs
# 224: 2542 GFLOPs
for i, resolution in enumerate([144]):
    args = {
        "wandb.tags": "[isic2019-resolution_scaling]",
        "training.dataset.resolution": resolution,
        "wandb.notes": f"r{i}",
    }
    run_command(args, global_args, test="instantiation")



depth_list = [16, 22]
resolution_list = [96, 144]


for depth in depth_list:
    for resolution in resolution_list:
        for i, width in enumerate([3, 5]):
            args = {
                "wandb.tags": "[isic2019-width_resolution_scaling]",
                "model.widen_factor": width,
                "model.depth": depth,
                "training.dataset.resolution": resolution,
                "wandb.notes": f"w{width}_d{depth}_r{resolution}",
            }
            # run_command(args, global_args, test="instantiation")
