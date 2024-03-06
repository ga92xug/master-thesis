#!/bin/bash


# 0.1
#$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet_pre \
#    train.dataset.reduction_factor=0.1 \
#    train.trainer.max_epochs=450 train.callbacks.early_stopping.patience=225 \
#    $PRE_TRAINED $SEEDS 

# 0.3
#$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet_pre \
#    train.dataset.reduction_factor=0.3 \
#    train.trainer.max_epochs=400 train.callbacks.early_stopping.patience=200 \
#    $PRE_TRAINED $SEEDS 

# 0.5
#$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet_pre \
#    train.dataset.reduction_factor=0.5 \
#    train.trainer.max_epochs=300 train.callbacks.early_stopping.patience=150 \
#    $PRE_TRAINED $SEEDS 

# 1
$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet_pre \
    train.dataset.reduction_factor=1 \
    train.trainer.max_epochs=150 train.callbacks.early_stopping.patience=75 \
    $PRE_TRAINED $SEEDS train/scheduler=none,cosine_annealing_lr