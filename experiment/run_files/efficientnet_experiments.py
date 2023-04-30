import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# experiment 1: efficientnet scaling laws

# global arguments
global_args = ["model=eq_wrn,wrn", "dataset=imagenette"]

# normal run
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=52", "model.widen_factor=1", "wandb.tags=[eff_exp_1]", "wandb.notes=normal_run"]
args.extend(global_args)
run_command(args)

# Scale WRN-52 by depth (d=4) 
# 52=1.259.194 -- *d =  -> 160=3.009.082; 124 -> 2.425.786
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=160,124", "model.widen_factor=1", "wandb.tags=[eff_exp_1]", "wandb.notes=depth_scaling"]
args.extend(global_args)
run_command(args)

# Scale WRN-52 by width (w=2)
# widen_factor 2: 3.080.634; 1.5: 2.452.602
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=52", "model.widen_factor=2,1.5", "wandb.tags=[eff_exp_1]", "wandb.notes=width_scaling"]
args.extend(global_args)
run_command(args)

# Scale WRN-52 by Resolution 
# 224 = 1.259.194 -- *r = 2.14 -> 3.061.434; r=1.83: 2.422.074
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=52", "model.widen_factor=1", "dataset.resolution_scaling=2.14,1.83", "wandb.tags=[eff_exp_1]", "wandb.notes=resolution_scaling"]
args.extend(global_args)
run_command(args)