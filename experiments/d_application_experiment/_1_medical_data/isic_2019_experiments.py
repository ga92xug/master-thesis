import os
import sys
sys.path.append(f"{os.getcwd()}")
from experiments.run_command import run_command

# experiment 1: depth or width scaling laws

# global arguments
global_args = {
    # wandb
    "wandb.project": "SL-Application",
    "wandb.tags": "[application, isic2019, medical_data]",
    # training
    "training": "isic2019-training",
    "training.dataset.resolution": 96,
    "training.dataset.batch_size": 32,
    "training.dataset.eval_batch_size": 32,
    "training.accumulate": 1,
    "training.epochs": 50,
}

# ECNN model
ecnn_args = {
    "model": "eq_wrn",
    "model.restrict": "[none,none]",
    "model.depth": 16,
    "model.widen_factor": 2,
}

# CNN
augmentation = {
    "RandomHorizontalFlip": {"p": 0.5},
    "Own_RandomRotation": {"degrees": [0,90,180,270]},
}

for augment in [augmentation]:
    cnn_args = {
        # model
        "model": "densenet",
        # augmentation
        "training.dataset.augment": augment,
        "training.epochs": 50*16,
        # wandb notes
        "wandb.notes": "densenet augmentation * 16",
    }
    run_command(cnn_args, global_args, test=False)


    


