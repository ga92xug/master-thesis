#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"
export PY_HPO="python experiments/d_application_experiment/_3_low_data_regime/low_data_HPO_new.py"

# 1
$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPs_isic2019=vit \
    wandb.tags=[isic2019_low_data_3] other.should_test=True \
    training.dataset.reduction_factor=1 \
    training.earlystop.stop=True \
    training.earlystop.patience=30 \
    training.earlystop.monitor=valid.acc_weighted \
    training.earlystop.mode=max \

    