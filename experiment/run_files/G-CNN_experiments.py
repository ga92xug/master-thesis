import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# experiment rotation

# global arguments
global_args = [
        "model=eq_wrn", 
        "training=cifar10-training",
        "wandb.tags=[G_CNN_exp1]",
        "wandb.group=G_CNN_exp1",
    ]

# 3x3
# 2.799.413
args = ["model.kernel_layout=[3,3]", "model.padding=1",
        "model.depth=16", "model.widen_factor=4"]
#run_command(args, global_args, test=False)

# 5x5
# 4.0= 7.155.141
# 2.5= 2.814.789
args = ["model.kernel_layout=[5,5]", "model.padding=2", 
        "model.depth=16", "model.widen_factor=2.5,4.0"]
#run_command(args, global_args, test=False)

# 7x7
# 4.0= 13.521.205
# 1.85= 2.756.293
args = ["model.kernel_layout=[7,7]", "model.padding=3", 
        "model.depth=16", "model.widen_factor=4.0"]
#run_command(args, global_args, test=False)


# global arguments
global_args = [
        "model=eq_wrn", 
        "training=cifar10-training",
        "wandb.tags=[G_CNN_exp1]",
        "wandb.group=G_CNN_exp1",
        "model.rotation=14",
    ]

# 3x3
# 2.799.413
args = ["model.kernel_layout=[3,3]", "model.padding=1",
        "model.depth=16", "model.widen_factor=4"]

#run_command(args, global_args, test=False)

# 5x5
# 4.0= 7.155.141
# 2.5= 2.814.789
args = ["model.kernel_layout=[5,5]", "model.padding=2", 
        "model.depth=16", "model.widen_factor=2.5"]

#run_command(args, global_args, test=False)

# 7x7
# 4.0= 13.521.205
# 1.85= 2.756.293
args = ["model.kernel_layout=[7,7]", "model.padding=3", 
        "model.depth=16", "model.widen_factor=1.85"]
#run_command(args, global_args, test=False)


# group experiment
global_args = [
        "model=eq_wrn", 
        "model.group=dihedral",
        "training=cifar10-training",
        "wandb.tags=[G_CNN_exp2]",
        "wandb.group=G_CNN_exp2",
    ]

# 3x3
args = ["model.depth=16", "model.widen_factor=4"]
run_command(args, global_args, test="instantiation")

args = ["model.depth=22", "model.widen_factor=6"]
run_command(args, global_args, test="instantiation")


# rotation experiment
global_args = [
        "model=eq_wrn", 
        "training=cifar10-training",
        "wandb.tags=[G_CNN_exp3]",
        "wandb.group=G_CNN_exp3",
    ]

# 3x3
args = ["model.kernel_layout=[3,3]", "model.padding=1", "model.group=cyclic",
        "model.depth=16", "model.widen_factor=4", "wandb.notes=rotation", 
        "model.rotation=12"]
#run_command(args, global_args, test=False)

args = ["model.kernel_layout=[5,5]", "model.padding=1", "model.group=cyclic",
        "model.depth=16", "model.widen_factor=4", "wandb.notes=rotation",
        "model.rotation=12"]
#run_command(args, global_args, test=False)


# restriction experiment
global_args = [
        "model=eq_wrn", 
        "training=cifar10-training",
        # wandb
        "wandb.tags=[G_CNN_exp4]",
        "wandb.group=G_CNN_exp4",
    ]

# 16_4
args = ["model.depth=16", "model.widen_factor=4", 
        "wandb.notes=16_4_restriction_exp", 
        "model.restrict=[invariant,invariant],[none,none],[halved,halved]"]
#run_command(args, global_args, test=False)
# 28_6
args = ["model.depth=28", "model.widen_factor=6", 
        "wandb.notes=28_6_restriction_exp", 
        "model.restrict=[invariant,invariant],[none,none],[halved,halved]"]
#run_command(args, global_args, test=False)
