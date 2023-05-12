import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command, run_command_test

# experiment rotation

# global arguments
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "wandb.tags=[exp_2_rotation]",
        "wandb.group=exp_2_rotation",
        "model.rotation=8,14",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1",
    ]

# 3x3
# 2.799.413
args = ["model.kernel_layout=[3,3]", "model.padding=1",
        "model.depth=16", "model.widen_factor=4"]
#run_command_test(args, global_args_test)
#run_command(args, global_args)

# 5x5
# 4.0= 7.155.141
# 2.5= 2.814.789
args = ["model.kernel_layout=[5,5]", "model.padding=2", 
        "model.depth=16", "model.widen_factor=2.5,4.0"]
#run_command_test(args, global_args_test)
run_command(args, global_args)

# 7x7
# 4.0= 13.521.205
# 1.85= 2.756.293
args = ["model.kernel_layout=[7,7]", "model.padding=3", 
        "model.depth=16", "model.widen_factor=1.85,4.0"]
#run_command_test(args, global_args_test)
run_command(args, global_args)
