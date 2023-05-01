import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# experiment 1: efficientnet scaling laws

# global arguments
global_args = ["model=eq_wrn,wrn", "dataset=imagenette"]

# normal run
# depth-width-resolution
# 52-1-224 => 1.259.194
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=52", "model.widen_factor=1", "wandb.tags=[eff_exp_1]", "wandb.notes=eq_wrn_baseline"]
run_command(args, global_args)

# Scale WRN-52 by depth 
# 124= 2.425.786
# 160= 3.009.082 
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=160,124", "model.widen_factor=1", "wandb.tags=[eff_exp_1]", "wandb.notes=depth_scaling"]
run_command(args, global_args)

# Scale WRN-52 by width
# 1.5= 2.452.602
# 2  = 3.080.634
# 
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=52", "model.widen_factor=2,1.5", "wandb.tags=[eff_exp_1]", "wandb.notes=width_scaling"]
run_command(args, global_args)

# Scale WRN-52 by Resolution 
# 1.83= 2.422.074
# 2.14= 3.061.434
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=52", "model.widen_factor=1", "dataset.resolution_scaling=2.14,1.83", "wandb.tags=[eff_exp_1]", "wandb.notes=resolution_scaling"]
run_command(args, global_args)


# eq_mobilenetv2
global_args = ["model=eq_mobilenetv2", "dataset=imagenette"]
# normal run
# depth-width-resolution
# 1-1-224 => 621.658
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=1", "model.width_multiplier=1", "wandb.tags=[eff_exp_1]", "wandb.notes=eq_mobilenetv2_baseline"]
run_command(args, global_args)

# Scale eq_mobilenetv2 by depth 
# d = 2 -> 1.380.554
# d = 4 -> 2.898.346
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=2,4", "model.width_multiplier=1", "wandb.tags=[eff_exp_1]", "wandb.notes=depth_scaling"]
run_command(args, global_args)

# Scale eq_mobilenetv2 by width
# width_multiplier 2: 1.331.050
# width_multiplier 3: 
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=1", "model.width_multiplier=2,1.5", "wandb.tags=[eff_exp_1]", "wandb.notes=width_scaling"]
run_command(args, global_args)

# Scale eq_mobilenetv2 by Resolution 
# 
# 224: r:1 = 621.658 
# 480: r:2.147 = 749.658
args = ["model.restrict=[null, invariant]", "model.padding=1", "model.depth_multiplier=1", "model.width_multiplier=1", "dataset.resolution_scaling=2.14,1.83", "wandb.tags=[eff_exp_1]", "wandb.notes=resolution_scaling"]
run_command(args, global_args)