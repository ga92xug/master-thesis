import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command
# global arguments
global_args = ["model=eq_wrn", "wandb.tags=[wrn_exp_5,additional]"]
"""
# done experiment depth 40 widen factor 0.5
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.depth=40", "model.widen_factor=0.5", "wandb.notes=wrn_exp_3_additional"]
args.extend(global_args)
run_command(args)

# more restriction
args = ["model.restrict=[halved, invariant]", "model.kernel_layout=[3,3]", "model.depth=40", "model.widen_factor=4", "wandb.notes=wrn_exp_3_additional_more_restriction"]
args.extend(global_args)
run_command(args)
"""
"""
# experiment 5 test rotations
# need to think about using 5x5 kernels with rotations larger than 4
# we already have rotation 4, so we just need to add 2, 8, 12, 16
# "wandb.notes=normal_run"
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=2", "wandb.notes=test_rotations", "model.rotation=4,10,12"]
run_command(args, global_args)

args = ["model.restrict=[null, invariant]", "model.kernel_layout=[5,5]", "model.padding=2", "model.depth=28", "model.widen_factor=2", "wandb.notes=test_rotations", "model.rotation=2,4,8,12,16"]
run_command(args, global_args)

# 2,4,8 done
args = ["model.restrict=[null, invariant]", "model.kernel_layout=[3,3]", "model.depth=16", "model.widen_factor=4", "wandb.notes=test_rotations", "model.rotation=12,16"]
run_command(args, global_args)

args = ["model.restrict=[null, invariant]", "model.kernel_layout=[5,5]", "model.padding=2", "model.depth=16", "model.widen_factor=4", "wandb.notes=test_rotations", "model.rotation=2,4,8,12,16"]
run_command(args, global_args)

# experiment 6 test groups
global_args = ["model=eq_wrn", "wandb.tags=[wrn_exp_6,additional]"]
# evaluate the effect of using different groups first 
# then evaluate the effect of using different groups with different rotations
"""
# test dihedral groups
args = ["model.restrict=[null, invariant]", "model.group=dihedral", "model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=2", "wandb.notes=test_dihedral", "model.rotation=4,8,10,12,16"]
run_command(args, global_args)

args = ["model.restrict=[halved, halved]", "model.group=dihedral", "model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=2", "wandb.notes=test_dihedral", "model.rotation=4,8,12"]
run_command(args, global_args)

args = ["model.restrict=[halved, halved]", "model.group=cyclic", "model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=2", "wandb.notes=test_cyclic", "model.rotation=4,8,12"]
run_command(args, global_args)

