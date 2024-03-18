#!/bin/bash
PY_SCRIPT="python src/main.py"
SEEDS="seed=0,1,2,3,4"

# 4 Blocks d=1-2-3-3 w=X r=128
$PY_SCRIPT -m experiment=scaling/all \
    train/network/blocks_args_dict=4blocks \
    train.network.blocks_args_dict._3.num_layers=3 \
    train.network.blocks_args_dict._4.num_layers=3 \
    train.network.width_coefficient=0.6,0.8,1.5,2.0 \
    train.dataset.resolution=128 \
    $SEEDS \

# 4 Blocks d=1-3-3-3 w=X r=192
$PY_SCRIPT -m experiment=scaling/all \
    train/network/blocks_args_dict=4blocks \
    train.network.blocks_args_dict._2.num_layers=3 \
    train.network.blocks_args_dict._3.num_layers=3 \
    train.network.blocks_args_dict._4.num_layers=3 \
    train.network.width_coefficient=0.6,0.8,1.5,2.0 \
    train.dataset.resolution=192 \
    $SEEDS \

