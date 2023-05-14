import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command, run_command_test

# python networks/eq_wrn.py -m model.kernel_layout=[3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]
# kernel_layout = [3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]

# global arguments
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.padding=1",
        "wandb.tags=[exp_1_eq_wrn]",
        "wandb.group=exp_1_eq_wrn",
        "model.rotation=8",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=1",
        "training.epochs=1",
    ]


# experiment 1 Type of convolutions in residual block
# params 2761013
args = ["model.kernel_layout=[3,3]", "model.depth=16", "model.widen_factor=4", "dataset.resolution=32", "wandb.notes=eq_wrn_baseline"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)

# params 2797573
args = ["model.kernel_layout=[1,3,1]", "model.depth=16", "model.widen_factor=5.2"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)

# 
args = ["model.kernel_layout=[3,1]", "model.depth=16", "model.widen_factor=5"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)


args = ["model.kernel_layout=[1,3]", "model.depth=16", "model.widen_factor=5"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)

# params 2707749
args = ["model.kernel_layout=[3,1,1]", "model.depth=16", "model.widen_factor=5"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)

# params 2932597
args = ["model.kernel_layout=[3,1,3]", "model.depth=16", "model.widen_factor=4"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)


# experiment 2 Number of convolutional layers per residual block
# global arguments
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.padding=1",
        "wandb.tags=[exp_2_eq_wrn]",
        "wandb.group=exp_2_eq_wrn",
        "model.rotation=8",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=1",
        "training.epochs=1",
    ]

# kernel_layout = [3]
args = ["model.kernel_layout=[3]", "model.depth=16", "model.widen_factor=4"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)
# kernel_layout = [3,3,3,3]
args = ["model.kernel_layout=[3,3,3,3]", "model.depth=16", "model.widen_factor=4"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)
# kernel_layout = [3,3,3] 
args = ["model.kernel_layout=[3,3,3]", "model.depth=16", "model.widen_factor=3.3"]
#run_command(args, global_args)
#run_command_test(args, global_args_test)


# experiment 3 Width of residual blocks

global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.padding=1",
        "wandb.tags=[exp_3_eq_wrn]",
        "wandb.group=exp_3_eq_wrn",
        "model.rotation=8",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=1",
        "training.epochs=1",
    ]

# d=40, k=1,2,4,8, [3,3]
args = ["model.depth=34", "model.widen_factor=1,2,4"]
#run_command(args, global_args)
# d=28, k=10,12 [3,3]
args = ["model.depth=28", "model.widen_factor=4,6", "wandb.notes=exp_3_eq_wrn"]
run_command(args, global_args)
# d=16,22, k=8, [3,3]
args = ["model.depth=22", "model.widen_factor=6,8", "wandb.notes=rot8"]
run_command(args, global_args)

# experiment 4 Dropout in residual blocks
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.padding=1",
        "wandb.tags=[exp_4_eq_wrn]",
        "wandb.group=exp_4_eq_wrn",
        "model.rotation=8",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=1",
        "training.epochs=1",
    ]
# d=16 k=4, [3,3], drop_out=0.3
args = ["model.depth=16", "model.widen_factor=4", "model.drop_out=0.3", "wandb.notes=exp_4_eq_wrn"]
#run_command(args, global_args)
# d=22 k=8, [3,3], drop_out=0.0,0.3
args = ["model.depth=22", "model.widen_factor=8", "model.drop_out=0.3", "wandb.notes=exp_4_eq_wrn"]
run_command(args, global_args)
# d=28 k=1, [3,3], drop_out=0.0,0.3
args = ["model.depth=28", "model.widen_factor=1", "model.drop_out=0.0,0.3", "wandb.notes=exp_4_eq_wrn"]
run_command(args, global_args)
