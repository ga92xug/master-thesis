#!/bin/bash

# 0.05
$PY_SCRIPT -m experiment=application/HPs_blood/eq_nasnet_pre \
    train.dataset.reduction_factor=0.05 \
    train.trainer.max_epochs=200 train.callbacks.early_stopping.patience=200 \
    $PRE_TRAINED $SEEDS 

# 0.1
$PY_SCRIPT -m experiment=application/HPs_blood/eq_nasnet_pre \
    train.dataset.reduction_factor=0.1 \
    train.trainer.max_epochs=150 train.callbacks.early_stopping.patience=150 \
    $PRE_TRAINED $SEEDS 

# 0.3
$PY_SCRIPT -m experiment=application/HPs_blood/eq_nasnet_pre \
    train.dataset.reduction_factor=0.3 \
    train.trainer.max_epochs=125 train.callbacks.early_stopping.patience=125 \
    $PRE_TRAINED $SEEDS 

# 0.5
$PY_SCRIPT -m experiment=application/HPs_blood/eq_nasnet_pre \
    train.dataset.reduction_factor=0.5 \
    train.trainer.max_epochs=100 train.callbacks.early_stopping.patience=100 \
    $PRE_TRAINED $SEEDS 

# 1
$PY_SCRIPT -m experiment=application/HPs_blood/eq_nasnet_pre \
    train.dataset.reduction_factor=1 \
    train.trainer.max_epochs=50 train.callbacks.early_stopping.patience=50 \
    $PRE_TRAINED $SEEDS 

