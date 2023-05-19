import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# python networks/eq_wrn.py -m model.kernel_layout=[3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]
# kernel_layout = [3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]

# global arguments
global_args = [
        # model
        "model=eq_wrn", 
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.padding=1",
        "model.rotation=8",
        # dataset
        "dataset=galaxy10",
        # training
        "training=galaxy-training",
        "optimizer=Adam",
        "optimizer.lr=0.01",
        "optimizer.weight_decay=1e-6",
        "optimizer.betas=[0.9,0.999]",
        # wandb
        "wandb.tags=[galaxy10]",
        "wandb.group=galaxy10",
    ]

# baseline
# flops = 2575G
args = ["model.depth=16", "model.widen_factor=4", "dataset.resolution=108", "wandb.notes=g10_baseline"]
run_command(args, global_args, test=False)

# width
# flops = 4091G
args = ["model.depth=16", "model.widen_factor=5.1", "dataset.resolution=108", "wandb.notes=g10_baseline"]
run_command(args, global_args, test=False)

# depth
# flops = 4039G
args = ["model.depth=22", "model.widen_factor=4", "dataset.resolution=108", "wandb.notes=g10_baseline"]
run_command(args, global_args, test=False)

# resolution
# flops = 4084G
args = ["model.depth=16", "model.widen_factor=4", "dataset.resolution=136", "wandb.notes=g10_baseline"]
run_command(args, global_args, test=False)
