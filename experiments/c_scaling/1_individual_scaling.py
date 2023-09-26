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
    "model": "eq_nasnet",
    "model.blocks_args_dict._1.skip": "conv",
    # training
    "training": "isic2019-training",
    "training.dataset.resolution": 96,
    "training.dataset.batch_size": 64,
    "training.dataset.eval_batch_size": 64,
    "training.accumulate": 1,
    # other 
    #"other.debug": True
}

# baseline
#  :  147 GFLOPs    
args = {
    "wandb.tags": "[isic2019-baseline]",
}
run_command(args, global_args, test=False)
quit()


# Depth 
# 1.5: 266 GFLOPs
# 2  : 347 GFLOPs
# 4  : 748 GFLOPs

for i, depth in enumerate([1.5,2,4]):
    args = {
        "wandb.tags": "[isic2019-depth_scaling]",
        "model.depth_coefficient": depth,
        "wandb.notes": f"d{i}",
    }
    run_command(args, global_args, test=False)

# Width
# 1.5: 
# 2:  512 GFLOPs
# 3:  1032 GFLOPs

# 4:  1917 GFLOPs
# 5:  3072 GFLOPs
for i, width in enumerate([1.5, 2, 2.5]):
    args = {
        "wandb.tags": "[isic2019-width_scaling]",
        "model.width_coefficient": width,
        "wandb.notes": f"w{i}",
    }
    run_command(args, global_args, test=False)


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
    run_command(args, global_args, test=False)

quit()

# odd even resolution
for seed in range(3):
    for i, resolution in enumerate([96, 95]):
        args = {
            "wandb.tags": f"[isic2019-resolution_odd, resolution-{resolution}]",
            "training.dataset.resolution": resolution,
            "other.seed": seed,
        }
        #run_command(args, global_args, test=False)