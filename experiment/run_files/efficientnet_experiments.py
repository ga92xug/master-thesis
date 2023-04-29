import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command
# global arguments
global_args = ["model=eq_wrn"]

# Scale ResNet-50 by depth (d=4) 
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=50", "model.widen_factor=1", "wandb.notes=eff_exp_1"]
args.extend(global_args)
run_command(args)

# Scale ResNet-50 by width (w=2)
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.padding=1", "model.depth=50", "model.widen_factor=2", "wandb.notes=eff_exp_1"]

# Resolution experiment 
# resolution experiment can only be done on imagenette. cifar10 is to small