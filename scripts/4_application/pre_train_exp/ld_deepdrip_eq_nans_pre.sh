#!/bin/bash


# 0.1
$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet_pre \
    train.dataset.reduction_factor=0.1 \
    train.trainer.max_epochs=150 train.callbacks.early_stopping.patience=100 \
    $PRE_TRAINED $SEEDS 

# 0.3
$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet_pre \
    train.dataset.reduction_factor=0.3 \
    train.trainer.max_epochs=120 train.callbacks.early_stopping.patience=80 \
    $PRE_TRAINED $SEEDS 

# 0.5
$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet_pre \
    train.dataset.reduction_factor=0.5 \
    train.trainer.max_epochs=100 train.callbacks.early_stopping.patience=60 \
    $PRE_TRAINED $SEEDS 

# 1
$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet_pre \
    train.dataset.reduction_factor=1 \
    train.trainer.max_epochs=50 train.callbacks.early_stopping.patience=30 \
    $PRE_TRAINED $SEEDS 