import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

"""
This file is solely used for quickly testing the networks
"""

# eq_wrn
# global arguments
global_args = [
        "model=eq_wrn", 
        "training=cifar10-training",
    ]


args = ["model.depth=16", "model.widen_factor=1"]
run_command(args, global_args, test=True)

# mobilenet
# global arguments
global_args = [
        "model=eq_mobilenetv2", 
        "training=cifar10-training",
        "model.fix_params_mode=heuristic",
    ]
args = []
run_command(args, global_args, test=True)
