#!/bin/bash
export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"

SEEDS="other.seed=0,1,2,3,4"

# 1
$PY_SCRIPT -m $SEEDS '+HPs_DeepDRiD=glob(*)' \
    wandb.tags=[DeepDRiD_low_data_3] other.should_test=True \
    training.dataset.reduction_factor=1 \

# 0.5
$PY_SCRIPT -m $SEEDS '+HPs_DeepDRiD=glob(*)' \
    wandb.tags=[DeepDRiD_low_data_3] other.should_test=True \
    training.dataset.reduction_factor=0.5 \
    training.epochs=300 training.earlystop.patience=80 \

# 0.3
$PY_SCRIPT -m $SEEDS '+HPs_DeepDRiD=glob(*)' \
    wandb.tags=[DeepDRiD_low_data_3] other.should_test=True \
    training.dataset.reduction_factor=0.3 \
    training.epochs=400 training.earlystop.patience=100 \

# 0.1
$PY_SCRIPT -m $SEEDS '+HPs_DeepDRiD=glob(*)' \
    wandb.tags=[DeepDRiD_low_data_3] other.should_test=True \
    training.dataset.reduction_factor=0.1 \
    training.epochs=450 training.earlystop.patience=250 \
