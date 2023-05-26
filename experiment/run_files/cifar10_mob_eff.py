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
        "model.restrict=[none,halved,invariant,none,none,none,none]",
        "model.min_feature_map_size=5",
        "model.kernel_size=3", 
        "model.padding=1",
        "wandb.tags=[eq_mobilenetv2]",
        "model.rotation=8",
    ]

# baseline
args = ["model.depth_multiplier=6", "model.width_multiplier=2.5", 
        "wandb.notes=eq_mobilenetv2_baseline",
        "dataset.resolution=32"]
run_command(args, global_args, test=False)


# efficientnet
# global arguments
global_args = [
        "model=eq_efficientnet", 
        "dataset=cifar10",
        "training=train_e2_100epochs",
        "optimizer=SGD",
        "model.fix_params_mode=no",
        "model.restrict=[none,halved,invariant,none,none,none,none]",
        "model.min_feature_map_size=5",
        "model.kernel_size=3", 
        "model.padding=1",
        "wandb.tags=[eq_mobilenetv2]",
        "model.rotation=8",
        "wandb.mode=disabled",
    ]

# baseline
# 1,1 param = 46.451; train time = 5.28
args = ["model.depth_multiplier=7", "model.width_multiplier=3.5", 
        "wandb.notes=eq_mobilenetv2_baseline",
        "dataset.resolution=32"]
#run_command(args, global_args, test="instantiation")
"""
# depth 
# x10 => 8.8 param = 422.366; train time = 13.04
# just scaling up depth is prohibitively expensive so smaller scale
# 40= 1.938.289 params; train time = 49.72
args = ["model.depth_multiplier=8.8,40", "model.width_multiplier=1", 
        "wandb.notes=eq_mobilenetv2 baseline",
        "dataset.resolution=32"]
run_command(args, global_args)

# width 
# x10 => 3.3 param = 447.651; train time = 14.55
# scale to 3.5 million params => 9.5 = 3.472.333; train time = 3.27
args = ["model.depth_multiplier=1", "model.width_multiplier=3.3,9.5", 
        "wandb.notes=eq_mobilenetv2 baseline",
        "dataset.resolution=32"]
run_command(args, global_args)


# depth and width
# x10 4.2,1.5 param = 399.405; train time = 143.5
args = ["model.depth_multiplier=4.2", "model.width_multiplier=1.5",
        "wandb.notes=eq_mobilenetv2 baseline",
        "dataset.resolution=32"]
run_command(args, global_args)

# scale to 3.5 million params => 9.5 = 3.472.333; train time = 3.27
# 8.5,3.1 = 3445032; train time = 12.72
args = ["model.depth_multiplier=8.5", "model.width_multiplier=3.1",
        "wandb.notes=eq_mobilenetv2 baseline",
        "dataset.resolution=32"]
run_command(args, global_args)
"""