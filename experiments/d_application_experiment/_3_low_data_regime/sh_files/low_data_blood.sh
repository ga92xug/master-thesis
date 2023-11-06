#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"
export PY_HPO="python experiments/d_application_experiment/_3_low_data_regime/low_data_HPO_new.py"

# 0.05
$PY_SCRIPT -m other.seed=0,1,2,3,4 '+exp_HPs_blood=glob(*)' \
    wandb.tags=[Blood_low_data] other.should_test=True \
    training.dataset.reduction_factor=0.05 \
    training.epochs=350 training.earlystop.patience=70


# 0.1
$PY_SCRIPT -m other.seed=0,1,2,3,4 '+exp_HPs_blood=glob(*)' \
    wandb.tags=[Blood_low_data] other.should_test=True \
    training.dataset.reduction_factor=0.1 \
    training.epochs=300 training.earlystop.patience=50

# 0.3
$PY_SCRIPT -m other.seed=0,1,2,3,4 '+exp_HPs_blood=glob(*)' \
    wandb.tags=[Blood_low_data] other.should_test=True \
    training.dataset.reduction_factor=0.3 \
    training.epochs=200 training.earlystop.patience=40


# 0.5
$PY_SCRIPT -m other.seed=0,1,2,3,4 '+exp_HPs_blood=glob(*)' \
    wandb.tags=[Blood_low_data] other.should_test=True \
    training.dataset.reduction_factor=0.5 \
    training.epochs=150 training.earlystop.patience=30


exit 0



# 1
$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPO_blood=efficientnet \
    wandb.tags=[Blood_low_data] other.should_test=True \
    training.dataset.reduction_factor=1