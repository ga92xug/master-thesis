#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"

# CIFAR-10
$PY_SCRIPT training=cifar10-training model.blocks_args_dict._3.se_ratio=0.0 model.dropout_rate=0.0 model.width_coefficient=1.5
# MNIST12k
$PY_SCRIPT training=mnist12k-training model.dropout_rate=0.0 model.blocks_args_dict._3.se_ratio=0.0
# MNIST-rot
$PY_SCRIPT training=mnist_rot-training \
model.blocks_args_dict._0.group=16 \
model.blocks_args_dict._1.group=16 \
model.blocks_args_dict._2.group=16 \
model.blocks_args_dict._3.group=16 \
model.blocks_args_dict._4.group=16 \
model.blocks_args_dict._3.se_ratio=0.0 \
model.dropout_rate=0.2 \
