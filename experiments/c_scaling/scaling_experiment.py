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
    "training.dataset.resolution": 96*2,
    # training
    "training": "isic2019-training",
    "training.dataset.batch_size": 32,
    "training.dataset.eval_batch_size": 32,
    "training.accumulate": 1,
}

# baseline
#  :  468 GFLOPs    
args = {
    "wandb.tags": "[isic2019-baseline]",
}
run_command(args, global_args, test="instantiation")

quit()
# odd even resolution
for seed in range(3):
    for i, resolution in enumerate([96, 95]):
        args = {
            "wandb.tags": f"[isic2019-resolution_odd, resolution-{resolution}]",
            "training.dataset.resolution": resolution,
            "other.seed": seed,
        }
        run_command(args, global_args, test=False)

# Depth 
# 22: 0730 GFLOPs
# 28: 0992 GFLOPs
# 34: 1254 GFLOPs
# 40: 1517 GFLOPs
# 46: 1779 GFLOPs
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
for i, resolution in enumerate([160, 224]):
    args = {
        "wandb.tags": "[isic2019-resolution_scaling]",
        "training.dataset.resolution": resolution,
        "wandb.notes": f"r{i}",
    }
    #run_command(args, global_args, test="instantiation")
