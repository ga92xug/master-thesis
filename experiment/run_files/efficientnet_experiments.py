import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command, run_command_test

# experiment 1: efficientnet scaling laws

# global arguments
global_args = [
        "model=eq_wrn", 
        "dataset=imagenette",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.kernel_layout=[3,3]", 
        "model.padding=1",
        "wandb.tags=[eff_exp_1]",
        "wandb.mode=disabled",
        "training.epochs=1",
        "model.rotation=8",
    ]

# normal run
# depth-width-resolution
# 108 is max resolution
# 58-1-108 => 1.024.069
args = ["model.depth=58", "model.widen_factor=1", "dataset.resolution=108", "wandb.notes=eq_wrn_baseline"]
run_command_test(args, global_args)

# Scale WRN-52 by depth 
# 130= 2.045.509
# 202 = 3.066.949 
args = ["model.depth=130,202", "model.widen_factor=1", "dataset.resolution=108", "wandb.notes=depth_scaling"]
run_command_test(args, global_args)

# Scale WRN-52 by width
# 1.47= 2.000.949 
# 1.85= 3.000.069
args = ["model.depth=58", "model.widen_factor=1.9", "dataset.resolution=108", "wandb.notes=width_scaling"]
run_command_test(args, global_args)

# Scale WRN-52 by Resolution 
# 360= 1.978.949 -> 2.25
# 480= 2.986.949 -> 3
args = ["model.depth=52", "model.widen_factor=1", "dataset.resolution=108", "wandb.notes=resolution_scaling"]
run_command_test(args, global_args)

"""
# eq_mobilenetv2
global_args = [
        "model=eq_mobilenetv2", 
        "dataset=imagenette",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.kernel_layout=[3,3]", 
        "model.padding=1",
        "wandb.tags=[eff_exp_1]",
        "wandb.mode=disabled",
        "training.epochs=1",
        "model.rotation=8",
    ]
# normal run
# depth-width-resolution
# 1-1-224 => 621.658
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=1", "model.width_multiplier=1", "wandb.tags=[eff_exp_1]", "wandb.notes=eq_mobilenetv2_baseline"]
run_command_test(args, global_args)

# Scale eq_mobilenetv2 by depth 
# d = 2 -> 1.380.554
# d = 4 -> 2.898.346
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=2,4", "model.width_multiplier=1", "wandb.tags=[eff_exp_1]", "wandb.notes=depth_scaling"]
run_command_test(args, global_args)

# Scale eq_mobilenetv2 by width
# width_multiplier 2: 1.331.050
# width_multiplier 3: 
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=1", "model.width_multiplier=2,1.5", "wandb.tags=[eff_exp_1]", "wandb.notes=width_scaling"]
run_command_test(args, global_args)

# Scale eq_mobilenetv2 by Resolution 
# 
# 224: r:1 = 621.658 
# 480: r:2.147 = 749.658
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=1", "model.width_multiplier=1", "dataset.resolution_scaling=2.14,1.83", "wandb.tags=[eff_exp_1]", "wandb.notes=resolution_scaling"]
run_command_test(args, global_args)

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