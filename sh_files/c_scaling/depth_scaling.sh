#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"

$PY_SCRIPT -m +exp_scaling=depth_scaling model.blocks_args_dict._3.se_ratio=0.0 model.skip=identity model.depth_coefficient=1.0,1.5
$PY_SCRIPT -m +exp_scaling=depth_scaling model.blocks_args_dict._3.se_ratio=0.0 model.skip=conv model.depth_coefficient=1.0,1.5