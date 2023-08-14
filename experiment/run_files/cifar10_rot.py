from pdb import run
import sys
sys.path.append('../run_files') # add parent directory
from run_command import run_command

global_args = []
args = [
        "training=cifar10-training",
        "training.dataset.rotation=True",
        "training.dataset.name=cifar10_rot",
        "other.verbose=5",
        "model.restrict=[halved,none],[none,none],[none,halved],[halved,halved]",
        "wandb.tags=[eq_wrn,baseline_cifar10_rot]",
    ]

run_command(args, global_args, test="instantiation")

