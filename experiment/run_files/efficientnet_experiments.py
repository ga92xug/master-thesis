import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command, run_command_test

# experiment 1: efficientnet scaling laws

# global arguments
global_args = [
        "model=eq_wrn", 
        "dataset=imagenette",
        "training=imagenette",
        "optimizer=Adam",
        #"model.fix_params_mode=heuristic",
        #"model.restrict=[halved,invariant]",
        #"model.kernel_layout=[3,3]", 
        #"model.padding=1",
        #"model.rotation=8",
        #"model.group=cyclic",
        "wandb.tags=[eff_exp_1]",
        #"training.augment_train=True",
    ]

# normal run
# depth-width-resolution
# 108 is max resolution
# 58-1-108 => 876.229, train time: 3.96
args = ["model.depth=16", "model.drop_out=0.3", "model.widen_factor=2", "dataset.resolution=224", "wandb.notes=eq_wrn_baseline"]
#run_command(args, global_args, test="instantiation")
#run_command_test(args, global_args_test)
"""
# Scale WRN by depth 
# 94= 1.386.949, train time: 6.098 
# 130 = 1897669, train time: 8.23
args = ["model.depth=94", "model.widen_factor=1", "dataset.resolution=108", "wandb.notes=depth_scaling"]
run_command(args, global_args)

# args = ["model.depth=130", "model.widen_factor=1", "dataset.resolution=108", "wandb.notes=depth_scaling"]
# run_command(args, global_args)

# Scale WRN by width
# 2.0= 3.310.333, train time: 5.92 
# 3.0= 7.263.045, train time: 7.97
args = ["model.depth=58", "model.widen_factor=2.0", "dataset.resolution=108", "wandb.notes=width_scaling"]
run_command(args, global_args)
# args = ["model.depth=58", "model.widen_factor=3.0", "dataset.resolution=108", "wandb.notes=width_scaling"]
# run_command(args, global_args)

# Scale WRN-52 by Resolution 
# 138= 953.029, train time: 6.10
# 160= 1.024.069, train time: 7.93
args = ["model.depth=58", "model.widen_factor=1", "dataset.resolution=138", "wandb.notes=resolution_scaling"]
run_command(args, global_args)
# args = ["model.depth=58", "model.widen_factor=1", "dataset.resolution=160", "wandb.notes=resolution_scaling"]
# run_command(args, global_args)
"""

# eq_mobilenetv2
global_args = [
        "model=eq_mobilenetv2", 
        "dataset=imagenette",
        "training=imagenette",
        "optimizer=Adam",
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant,null,null,null,null,null]",
        "model.kernel_size=3", 
        "model.padding=1",
        "wandb.tags=[eff_exp_1]",
        "model.rotation=8", 
    ]

# normal run
# depth-width-resolution
# 1-1-224 => 621.658
args = ["model.depth_multiplier=1", "model.width_multiplier=1", "dataset.resolution=108", "wandb.notes=eq_mobilenetv2_baseline"]
#run_command(args, global_args, test="instantiation")
#run_command_test(args, global_args_test)
"""
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

# eq_efficientnet
global_args = [
        "model=eq_efficientnet", 
        "dataset=imagenette",
        "training=imagenette",
        "optimizer=Adam",
        #"model.fix_params_mode=heuristic",
        #"model.restrict=[null,null,null,null,null,halved,invariant,null]",
        "wandb.tags=[eff_exp_1]",
        
        #"model.rotation=8", 
    ]

# normal run
# depth-width-resolution
# 1-1-224 => 5.787.953
args = ["model.global_params.depth_coefficient=0.7", "model.global_params.width_coefficient=0.7",
        "dataset.resolution=224", 
        "wandb.tags=[eff_exp_1]", "wandb.notes=eq_efficientnet_baseline"]
run_command(args, global_args, test="instantiation")

# Scale eq_efficientnet by depth
# d = 1.1 -> 9.359.204

# Scale eq_efficientnet by width
# w = 1.275 -> 9.447.196

# Scale eq_efficientnet by Resolution
