import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

"""
This file is solely used for quickly testing the networks
"""

# global arguments
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        #"model.restrict=[reflection,invariant]",
        #"model.kernel_layout=[3,3]", 
        #"model.group=dihedral",
        #"model.padding=1",
        #"model.rotation=4",
        #"model.bias=True",
        #"other.verbose=1",
    ]


# 3x3
# 2.799.413
args = [#"model.depth=16", "model.widen_factor=1", 
        "wandb.notes=check_if_train_time_log",
        "dataset.resolution=32", "model.drop_out=0.0"]
run_command(args, global_args, test=True)

# mobilenet

# global arguments
global_args = [
        "model=eq_mobilenetv2", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        #"model.restrict=[halved,invariant]",
        "model.padding=1",
        "model.rotation=8",
        "other.verbose=1"
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1000",
    ]


args = [# "model.depth=16", "model.widen_factor=1", 
        "wandb.notes=check_if_train_time_log",
        "dataset.resolution=224", "model.drop_out=0.0"]
#run_command_test(args, global_args_test)
#run_command(args, global_args_test)



# mobilenet

# global arguments
global_args = [
        "model=eq_efficientnet", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        #"model.restrict=[halved,invariant]",
        "model.rotation=8",
        "other.verbose=1"
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1000",
    ]


args = [# "model.depth=16", "model.widen_factor=1", 
        "wandb.notes=check_if_train_time_log",
        "dataset.resolution=224", "model.global_params.drop_out=0.0"]
#run_command_test(args, global_args_test)
#run_command(args, global_args_test)