import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# experiment rotation

# global arguments
global_args = ["model=eq_wrn"]


args = ["model.restrict=[halved, halved]", "model.kernel_layout=[5,5]", "model.padding=2", "model.depth=16", "model.widen_factor=4", "wandb.tags=[rotation_exp_1]", "wandb.notes=rot8"]
run_command(args, global_args)

args = ["model.restrict=[halved, halved]", "model.kernel_layout=[5,5]", "model.padding=2", "model.depth=28", "model.widen_factor=7", "wandb.tags=[rotation_exp_1]", "wandb.notes=28_7rot8"]
run_command(args, global_args)