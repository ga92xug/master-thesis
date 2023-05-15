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
        "wandb.tags=[G_CNN_exp1]",
        "wandb.group=G_CNN_exp1",
        "model.rotation=8",
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
#run_command(args, global_args)

# 7x7
# 4.0= 13.521.205
# 1.85= 2.756.293
args = ["model.kernel_layout=[7,7]", "model.padding=3", 
        "model.depth=16", "model.widen_factor=4.0"]
#run_command_test(args, global_args_test)
#run_command(args, global_args)


# global arguments
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "wandb.tags=[G_CNN_exp1]",
        "wandb.group=G_CNN_exp1",
        "model.rotation=14",
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
        "model.depth=16", "model.widen_factor=2.5"]
#run_command_test(args, global_args_test)
#run_command(args, global_args)

# 7x7
# 4.0= 13.521.205
# 1.85= 2.756.293
args = ["model.kernel_layout=[7,7]", "model.padding=3", 
        "model.depth=16", "model.widen_factor=1.85"]
#run_command_test(args, global_args_test)
#run_command(args, global_args)


# group experiment
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "wandb.tags=[G_CNN_exp2]",
        "wandb.group=G_CNN_exp2",
        "model.rotation=8",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1",
    ]

# 3x3
args = ["model.kernel_layout=[3,3]", "model.padding=1", "model.group=dihedral",
        "model.depth=16", "model.widen_factor=4", "wandb.notes=group_dihedral"]
#run_command_test(args, global_args_test)
run_command(args, global_args, test=False)

args = ["model.kernel_layout=[3,3]", "model.padding=1", "model.group=dihedral",
        "model.depth=22", "model.widen_factor=6", "wandb.notes=group_dihedral"]
run_command(args, global_args, test=False)


# rotation experiment
global_args = [
        "model=eq_wrn", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "wandb.tags=[G_CNN_exp3]",
        "wandb.group=G_CNN_exp3",
        "model.rotation=8",
    ]

# 3x3
args = ["model.kernel_layout=[3,3]", "model.padding=1", "model.group=cyclic",
        "model.depth=16", "model.widen_factor=4", "wandb.notes=rotation", 
        "model.rotation=2,10,16,20"]
#run_command_test(args, global_args_test)
run_command(args, global_args, test=False)

args = ["model.kernel_layout=[5,5]", "model.padding=1", "model.group=cyclic",
        "model.depth=16", "model.widen_factor=4", "wandb.notes=rotation",
        "model.rotation=2,10,16,20"]
run_command(args, global_args, test=False)
