import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# experiment 1: efficientnet scaling laws

# global arguments
global_args = ["model=eq_wrn", "dataset=cifar10", "training=train_e2_100epochs"]

# normal run
args = ["model.kernel_layout=[3,3],[5,5],[7,7]", "wandb.tags=[convergence_train_e2_100epochs_2]"]
run_command(args, global_args)