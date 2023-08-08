import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# global arguments
global_args = [
        "model=eq_nasnet", 
        "training=cifar10-training",
        "wandb.tags=[nas_result_test]",
    ]


args = []
run_command(args, global_args, test="instantiation")

