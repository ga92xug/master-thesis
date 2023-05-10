import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command, run_command_test

# experiment 1: depth or width scaling laws

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
        "wandb.tags=[depth_width_scaling]",
        "model.rotation=8",
    ]

global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=100",
        "training.epochs=1",
    ]

# normal run
# depth-width-resolution
# 3.6s per 100 steps
args = ["model.depth=16", "model.widen_factor=4", "dataset.resolution=32", "wandb.notes=eq_wrn_baseline"]
#run_command(args, global_args)
# run_command_test(args, global_args_test)

# Depth 
# 5.9s per 100 steps
args = ["model.depth=28", "model.widen_factor=4", "dataset.resolution=32", "wandb.notes=depth_scaling"]
run_command(args, global_args)
#run_command_test(args, global_args_test)

# Width
# 5.6s per 100 steps
args = ["model.depth=16", "model.widen_factor=6.5", "dataset.resolution=32", "wandb.notes=width_scaling"]
run_command(args, global_args)
#run_command_test(args, global_args_test)

# combound scaling
# 6.0s per 100 steps
args = ["model.depth=22", "model.widen_factor=5", "dataset.resolution=32", "wandb.notes=width_scaling"]
run_command(args, global_args)
#run_command_test(args, global_args_test)

"""
# eq_mobilenetv2
global_args = [
        "model=eq_mobilenetv2", 
        "dataset=imagenette",
        "training=imagenette",
        "model.fix_params_mode=heuristic",
        "model.restrict=[none,none,none,none,none,halved,invariant]",
        "model.kernel_size=3", 
        "model.padding=1",
        "wandb.tags=[eff_exp_1]",
        "model.rotation=8", 
    ]
global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1",
    ]
# normal run
# depth-width-resolution
# 1-1-224 => 621.658
args = ["model.depth_multiplier=1", "model.width_multiplier=1", "dataset.resolution=224", "wandb.notes=eq_mobilenetv2_baseline"]
run_command(args, global_args)
#run_command_test(args, global_args_test)

# Scale eq_mobilenetv2 by depth 
# d = 2 -> 1.380.554
# d = 4 -> 2.898.346
args = ["model.depth_multiplier=1.8", "model.width_multiplier=1", "dataset.resolution=224", "wandb.notes=depth_scaling"]
#run_command_test(args, global_args_test)
run_command(args, global_args)

# Scale eq_mobilenetv2 by width
# width_multiplier 2: 1.331.050
# width_multiplier 3: 
args = ["model.depth_multiplier=1", "model.width_multiplier=1.4", "dataset.resolution=224", "wandb.notes=width_scaling"]
#run_command_test(args, global_args_test)
run_command(args, global_args)

# Scale eq_mobilenetv2 by Resolution 
# 
# 224: r:1 = 621.658 
# 480: r:2.147 = 749.658
args = ["model.depth_multiplier=1", "model.width_multiplier=1", "dataset.resolution=224", "wandb.notes=resolution_scaling"]
#run_command_test(args, global_args_test)
"""
"""
# eq_efficientnet
global_args = ["model=eq_efficientnet", "dataset=imagenette"]

# normal run
# depth-width-resolution
# 1-1-224 => 5.787.953
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=1", "model.width_multiplier=1", "wandb.tags=[eff_exp_1]", "wandb.notes=eq_efficientnet_baseline"]
run_command_test(args, global_args)

# Scale eq_efficientnet by depth
# d = 1.1 -> 9.359.204

# Scale eq_efficientnet by width
# w = 1.275 -> 9.447.196

# Scale eq_efficientnet by Resolution
"""