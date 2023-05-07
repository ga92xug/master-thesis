import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

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
        "model.kernel_layout=[3,3]", 
        "model.padding=1",
        "wandb.tags=[rotation_exp_2]",
        "model.rotation=8",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1",
    ]

# to do control for param size
args = ["model.kernel_layout=[3,3]", "model.padding=1", "model.rotation=8,12,16",
        "model.depth=16", "model.widen_factor=4", "wandb.notes=kernel3x3"]
run_command(args, global_args)

args = ["model.kernel_layout=[5,5]", "model.padding=2", "model.rotation=8,12,16",
        "model.depth=16", "model.widen_factor=4", "wandb.notes=kernel5x5"]
run_command(args, global_args)

args = ["model.kernel_layout=[7,7]", "model.padding=3", "model.rotation=8,12,16",
        "model.depth=16", "model.widen_factor=4", "wandb.notes=kernel7x7"]
run_command(args, global_args)

# args = ["model.restrict=[halved, halved]", "model.kernel_layout=[5,5]", "model.padding=2", "model.depth=28", "model.widen_factor=7", "wandb.tags=[rotation_exp_1]", "wandb.notes=28_7rot8"]
# run_command(args, global_args)


