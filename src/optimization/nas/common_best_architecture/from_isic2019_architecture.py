import os
import sys
sys.path.append(f"{os.getcwd()}")
from experiments.run_command import run_command

# experiment 1: depth or width scaling laws

global_args = {
    # wandb
    "wandb.project": "SL-NAS-common-best",
    #"wandb.tags": "[application, isic2019, medical_data]",
    # training
    #"training": "isic2019-training",
    #"training.dataset.resolution": 96,
    #"training.dataset.batch_size": 32,
    #"training.dataset.eval_batch_size": 32,
    #"training.accumulate": 1,
    #"training.epochs": 50,
}

# ECNN model
args = {
    "model": "eq_nasnet",
}

# impact of SE



# impact of dropout
