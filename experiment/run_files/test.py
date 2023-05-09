import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command, run_command_test

# experiment rotation
# experiment 2 discretization artifacts

# global arguments
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.kernel_size=3", 
        "model.padding=1",
        "wandb.tags=[rotation_exp_2]",
        "model.rotation=8",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1",
    ]

# 3x3
# 2.799.413
args = ["model.depth=10", "model.widen_factor=1.5", 
        "wandb.notes=kernel3x3",
        "dataset.resolution=32"]
run_command_test(args, global_args_test)
#run_command(args, global_args)