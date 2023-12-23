#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"


# kernel layout
$PY_SCRIPT -m +experiments=initial \
    model.kernel_layout=[1,3,1] \
    model.widen_factor=5.2 \
    wandb.tags=[kernel_layout]

$PY_SCRIPT -m +experiments=initial \
    model.kernel_layout=[3,1],[1,3],[3,1,1] \
    model.widen_factor=5 \
    wandb.tags=[kernel_layout]

$PY_SCRIPT -m +experiments=initial \
    model.kernel_layout=[3,1,3] \
    model.widen_factor=4 \
    wandb.tags=[kernel_layout]

# Number of convolutional layers per residual block
$PY_SCRIPT -m +experiments=initial \
    model.widen_factor=1 \
    model.kernel_layout=[3,3,3,3] \
    model.depth=22 \
    wandb.tags=[kernel_layout]

$PY_SCRIPT -m +experiments=initial \
    model.widen_factor=1 \
    model.kernel_layout=[3,3,3] \
    model.depth=28 \
    wandb.tags=[kernel_layout]

$PY_SCRIPT -m +experiments=initial \
    model.widen_factor=1 \
    model.kernel_layout=[3,3] \
    model.depth=40 \
    wandb.tags=[kernel_layout]

$PY_SCRIPT -m +experiments=initial \
    model.widen_factor=1 \
    model.kernel_layout=[3] \
    model.depth=76 \
    wandb.tags=[kernel_layout]
