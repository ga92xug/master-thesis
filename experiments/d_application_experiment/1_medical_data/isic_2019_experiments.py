import os
from pyexpat import model
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
}

# ECNN model
ecnn_args = {
    "model": "eq_wrn",
    "model.restrict": "[none,none]",
    "model.depth": 16,
    "model.widen_factor": 2,
}

# CNN
cnn_args = {
    # model
    "model": "densenet",
    # augmentation
    "training.dataset.augment": {
        #"RandomHorizontalFlip": {"p": 0.5},
        "Own_RandomRotation": {"degrees": [0,90,180,270]},
    },

}
run_command(cnn_args, global_args, test=True)


    


