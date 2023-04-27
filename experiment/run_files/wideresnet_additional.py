from . import run_command
# global arguments
global_args = ["model=eq_wrn", "model.restrict=invariant"]

# experiment depth 40 widen factor 0.5
args = ["model.kernel_layout=[3,3]", "model.depth=40", "model.widen_factor=0.5", "wandb.notes=wrn_exp_3_additional"]
run_command(global_args.extend(args))

# more restriction