#!/bin/bash
SEEDS="seed=0,1,2"

# 0.1
$PY_SCRIPT -m experiment=application/HPs_DeepDRiD/eq_nasnet \
    train.dataset.reduction_factor=0.1 \
    train.trainer.max_epochs=80 train.callbacks.early_stopping.patience=40 \
    $PRE_TRAINED $SEEDS train.network.dropout_rate=0.5,0.2

# 0.3
$PY_SCRIPT experiment=application/HPs_DeepDRiD/eq_nasnet \
    train.dataset.reduction_factor=0.3 \
    train.trainer.max_epochs=60 train.callbacks.early_stopping.patience=30 \
    $PRE_TRAINED $SEEDS train.network.dropout_rate=0.5,0.2

# 0.5
$PY_SCRIPT experiment=application/HPs_DeepDRiD/eq_nasnet \
    train.dataset.reduction_factor=0.5 \
    train.trainer.max_epochs=50 train.callbacks.early_stopping.patience=25 \
    $PRE_TRAINED $SEEDS train.network.dropout_rate=0.5,0.2

# 1
$PY_SCRIPT experiment=application/HPs_DeepDRiD/eq_nasnet \
    train.dataset.reduction_factor=1 \
    train.trainer.max_epochs=30 train.callbacks.early_stopping.patience=15 \
    $PRE_TRAINED $SEEDS train.network.dropout_rate=0.5,0.2