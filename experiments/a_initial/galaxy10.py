import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# global arguments
global_args = [
        # model
        "model=eq_wrn", 
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.rotation=8",
        # training
        "training=galaxy-training",
        # wandb
        "wandb.tags=[galaxy10]",
        "wandb.group=galaxy10",
    ]

# baseline
# flops = 2575G
args = ["model.depth=16", "model.widen_factor=4", "dataset.resolution=108"]
run_command(args, global_args, test=False)

# width
# flops = 4091G
args = ["model.depth=16", "model.widen_factor=5.1", "dataset.resolution=108"]
run_command(args, global_args, test=False)

# depth
# flops = 4039G
args = ["model.depth=22", "model.widen_factor=4", "dataset.resolution=108"]
run_command(args, global_args, test=False)

# resolution
# flops = 4084G
args = ["model.depth=16", "model.widen_factor=4", "dataset.resolution=136"]
run_command(args, global_args, test=False)
