#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"

SEEDS="other.seed=0,1,2,3,4"

# 4 Blocks d=1-2-3-3 w=X r=128
$PY_SCRIPT -m +experiments=scaling \
    model.blocks_args_dict=4blocks \
    model.blocks_args_dict._3.num_layers=3 \
    model.blocks_args_dict._4.num_layers=3 \
    model.width_coefficient=0.6,0.8,1.5,2.0 \
    training.dataset.resolution=128 \
    $SEEDS \

# 4 Blocks d=1-3-3-3 w=X r=192
$PY_SCRIPT -m +experiments=scaling \
    model.blocks_args_dict=4blocks \
    model.blocks_args_dict._2.num_layers=3 \
    model.blocks_args_dict._3.num_layers=3 \
    model.blocks_args_dict._4.num_layers=3 \
    model.width_coefficient=0.6,0.8,1.5,2.0 \
    training.dataset.resolution=192 \
    $SEEDS \

