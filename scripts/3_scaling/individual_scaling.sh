#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"

SEEDS="other.seed=0,1,2,3,4"


# baseline 3 blocks and 4 blocks
$PY_SCRIPT -m +experiments=scaling \
    model.blocks_args_dict=3blocks,4blocks \
    $SEEDS \

# width
$PY_SCRIPT -m +experiments=scaling \
    model.width_coefficient=1.25,1.5,1.75,2.0,2.5,3.0 \
    $SEEDS \

# resolution
$PY_SCRIPT -m +experiments=scaling \
    model.blocks_args_dict=3blocks,4blocks \
    training.dataset.resolution=96,128,192,224 \
    $SEEDS \
    
# depth
# 3 blocks
$PY_SCRIPT -m +experiments=scaling \
    model.blocks_args_dict=3blocks \
    model.blocks_args_dict._3.num_layers=3 \
    $SEEDS \

$PY_SCRIPT -m +experiments=scaling \
    model.blocks_args_dict=3blocks \
    model.blocks_args_dict._2.num_layers=3 \
    model.blocks_args_dict._3.num_layers=4 \
    $SEEDS \

# 4 blocks
$PY_SCRIPT -m +experiments=scaling \
    model.blocks_args_dict=4blocks \
    model.blocks_args_dict._3.num_layers=3 \
    model.blocks_args_dict._4.num_layers=3 \
    $SEEDS \

$PY_SCRIPT -m +experiments=scaling \
    model.blocks_args_dict=4blocks \
    model.blocks_args_dict._2.num_layers=3 \
    model.blocks_args_dict._3.num_layers=4 \
    model.blocks_args_dict._4.num_layers=4 \
    $SEEDS \