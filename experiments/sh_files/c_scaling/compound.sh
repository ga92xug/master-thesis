#!/bin/bash

SEEDS="other.seed=0,1,2"
TAGS="wandb.tags=[constant_scheduler,compound_scaling]"

# b3 d1 r128 
$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128 \
    model.width_coefficient=1.5,2.0 \
    model.depth_coefficient=1 \
    $TAGS

# b3 d1.4 r96
# submitted 
$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=96 \
    model.width_coefficient=1.5,2.0 \
    model.depth_coefficient=1.4 \
    $TAGS

# b4 d1 r96
# submitted
$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=96 \
    model.width_coefficient=1.5,2.0 \
    model.depth_coefficient=1 \
    model/blocks_args_dict=4blocks \
    $TAGS

# b4 d1 r128
# submitted
$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128 \
    model.width_coefficient=1.5,2.0 \
    model.depth_coefficient=1 \
    model/blocks_args_dict=4blocks \
    $TAGS