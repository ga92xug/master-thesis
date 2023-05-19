import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

# python networks/eq_wrn.py -m model.kernel_layout=[3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]
# kernel_layout = [3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]

# global arguments
global_args = [
        "model=eq_wrn", 
        "model.fix_params_mode=heuristic",
        "model.restrict=[halved,invariant]",
        "model.padding=1",
        "model.rotation=8",

        "dataset=galaxy10",
        "dataset.resolution=108",

        "training=train_e2_100epochs",
        "optimizer=SGD",
        
        "wandb.tags=[galaxy10]",
        "wandb.group=galaxy10",
    ]


args = ["model.kernel_layout=[3,3]", "model.depth=16", "model.widen_factor=4", "wandb.notes=g10_baseline"]
run_command(args, global_args, test=False)