import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# global arguments
global_args = [
        # model
        "model=eq_wrn", 
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.padding=1",
        "model.rotation=8",
        # dataset
        "dataset=mnist_rot",
        # training
        "training=mnist_rot",
        "optimizer=Adam",
        "optimizer.lr=0.015",
        # wandb
        "wandb.tags=[mnist_rot, baseline_mnist_rot]",
        "wandb.group=mnist_rot",
    ]

# baseline
# flops = 2575G
args = ["model.depth=16", "model.widen_factor=4", "wandb.notes=baseline_mnist_rot"]
run_command(args, global_args, test=False)