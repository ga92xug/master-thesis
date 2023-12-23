#!/bin/bash
export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"

SEEDS="other.seed=0,1,2,3,4"

# 1
$PY_SCRIPT -m $SEEDS '+HPs_isic2019=glob(*)' \
    wandb.tags=[isic2019_low_data_3] \
    training.dataset.reduction_factor=1 \
    training.earlystop.stop=True \
    training.earlystop.monitor=valid.acc_weighted \
    training.earlystop.mode=max \
    training.earlystop.patience=30 \
    