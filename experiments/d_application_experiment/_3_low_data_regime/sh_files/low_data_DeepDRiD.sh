#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"
export PY_HPO="python experiments/d_application_experiment/_3_low_data_regime/low_data_HPO_new.py"


# 0.5
$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPs_DeepDRiD=vit,efficientnet_pre,eq_nasnet \
    wandb.tags=[DeepDRiD_low_data_2] other.should_test=True \
    training.dataset.reduction_factor=0.5 \
    training.epochs=300 training.earlystop.patience=80 \
    training.earlystop.monitor=valid.acc \
    training.earlystop.mode=max \
    #other.debug=True

exit 0

# 0.3
$PY_SCRIPT -m other.seed=0,1,2,3,4 '+exp_HPs_DeepDRiD=glob(*)' \
    wandb.tags=[DeepDRiD_low_data_2] other.should_test=True \
    training.dataset.reduction_factor=0.3 \
    training.epochs=400 training.earlystop.patience=100

# 0.1
$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPs_DeepDRiD=efficientnet,efficientnet_pre,vit,vit_pre \
    wandb.tags=[DeepDRiD_low_data_2] other.should_test=True \
    training.dataset.reduction_factor=0.1 \
    training.epochs=450 training.earlystop.patience=250

# 1
$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPs_DeepDRiD=efficientnet,vit \
    wandb.tags=[DeepDRiD_low_data_2] other.should_test=True \
    training.dataset.reduction_factor=1 \
    training.earlystop.patience=50
