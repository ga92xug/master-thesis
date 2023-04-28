import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command
# global arguments
global_args = ["model=eq_wrn"]
"""
# experiment depth 40 widen factor 0.5
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.depth=40", "model.widen_factor=0.5", "wandb.notes=wrn_exp_3_additional"]
args.extend(global_args)
run_command(args)

# more restriction
args = ["model.restrict=[halved, invariant]", "model.kernel_layout=[3,3]", "model.depth=40", "model.widen_factor=4", "wandb.notes=wrn_exp_3_additional_more_restriction"]
args.extend(global_args)
run_command(args)

# experiment 5 test rotations
# need to think about using 5x5 kernels with rotations larger than 4
# we already have rotation 4, so we just need to add 2, 8, 12, 16
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=2", "wandb.notes=wrn_exp_5_additional_test_rotations", "model.rotation=2,8"]
args.extend(global_args)
run_command(args)
"""
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[5,5]", "model.padding=2", "model.depth=28", "model.widen_factor=2", "wandb.notes=wrn_exp_5_additional_test_rotations", "model.rotation=8,12,16"]
args.extend(global_args)
run_command(args)

# args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.depth=16", "model.widen_factor=4", "wandb.notes=wrn_exp_5_additional_test_rotations", "model.rotation=2,8"]
# args.extend(global_args)
# run_command(args)

args = ["model.restrict=[null, invariant]", "model.kernel_layout=[5,5]", "model.padding=2", "model.depth=16", "model.widen_factor=4", "wandb.notes=wrn_exp_5_additional_test_rotations", "model.rotation=8,12,16"]
args.extend(global_args)
run_command(args)

# experiment 6 test groups