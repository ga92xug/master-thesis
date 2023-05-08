import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command, run_command_test

# experiment rotation
# experiment 2 discretization artifacts

# global arguments
global_args = [
        "model=eq_mobilenetv2", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[none,halved,invariant]",
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
args = ["model.depth_multiplier=40", "model.width_multiplier=2.4", 
        "wandb.notes=kernel3x3",
        "dataset.resolution=108"]
run_command_test(args, global_args_test)
#run_command(args, global_args)