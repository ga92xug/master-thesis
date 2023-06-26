import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command, run_command_test

# experiment 1: depth or width scaling laws

# global arguments
global_args = [
        "model=eq_wrn", 
        "training=cifar10-training",
        "wandb.tags=[depth_width_scaling]",
        "wandb.group=depth_width_scaling",
    ]


# normal run
# depth-width-resolution
# 3.4s per 100 steps
args = ["model.depth=16", "model.widen_factor=4", "wandb.notes=eq_wrn_baseline"]
#run_command(args, global_args)

# Depth 
# 5.8s per 100 steps
args = ["model.depth=28", "model.widen_factor=4", "wandb.notes=depth_scaling"]
#run_command(args, global_args)

# Width
# 5.4s per 100 steps
args = ["model.depth=16", "model.widen_factor=6.5", "wandb.notes=width_scaling"]
#run_command(args, global_args)

# combound scaling
# 5.8s per 100 steps
args = ["model.depth=22", "model.widen_factor=5", "wandb.notes=width_scaling"]
#run_command(args, global_args)
