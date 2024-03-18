#!/bin/bash
PY_SCRIPT="python src/main.py"
SEEDS="seed=0,1,2,3,4"


# baseline 3 blocks and 4 blocks
$PY_SCRIPT -m experiment=scaling/all \
    train/network/blocks_args_dict=3blocks,4blocks \
    $SEEDS \

# width
$PY_SCRIPT -m experiment=scaling/all \
    train.network.width_coefficient=1.25,1.5,1.75,2.0,2.5,3.0 \
    $SEEDS \

# resolution
$PY_SCRIPT -m experiment=scaling/all \
    train/network/blocks_args_dict3blocks,4blocks \
    train.dataset.resolution=96,128,192,224 \
    $SEEDS \
    
# depth
# 3 blocks
$PY_SCRIPT -m experiment=scaling/all \
    train/network/blocks_args_dict=3blocks \
    train.network.blocks_args_dict._3.num_layers=3 \
    $SEEDS \

$PY_SCRIPT -m experiment=scaling/all \
    train/network/blocks_args_dict=3blocks \
    train.network.blocks_args_dict._2.num_layers=3 \
    train.network.blocks_args_dict._3.num_layers=4 \
    $SEEDS \

# 4 blocks
$PY_SCRIPT -m experiment=scaling/all \
    train/network/blocks_args_dict=4blocks \
    train.network.blocks_args_dict._3.num_layers=3 \
    train.network.blocks_args_dict._4.num_layers=3 \
    $SEEDS \

$PY_SCRIPT -m experiment=scaling/all \
    train/network/blocks_args_dict=4blocks \
    train.network.blocks_args_dict._2.num_layers=3 \
    train.network.blocks_args_dict._3.num_layers=4 \
    train.network.blocks_args_dict._4.num_layers=4 \
    $SEEDS \