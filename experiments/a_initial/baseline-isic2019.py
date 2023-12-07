import os
import sys
sys.path.append(f"{os.getcwd()}")
from experiments.run_command import run_command

global_args = [
    "training=isic2019-training",
]
args = [
    "model=eq_wrn",
    "model.restrict=[none,none]",
    "model.drop_out=0.5",

    "wandb.project=SL-baselines",
    "wandb.tags=[eq_wrn,isic2019]",
]

run_command(args, global_args, test=False)

args = [
    "model=densenet",
    "model.drop_rate=0.3",

    "wandb.project=SL-baselines",
    "wandb.tags=[densenet,isic2019]",
]
run_command(args, global_args, test=False)