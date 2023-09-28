#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"
SEEDS="other.seed=0,1,2"

# baseline
# 72 GFLOPs
# $PY_SCRIPT -m +exp_scaling=general $SEEDS

# width_coefficient
# 1.5 = 149 GFLOPs
# 2.0 = 252 GFLOPs
# 2.5 = 381 GFLOPs
# $PY_SCRIPT -m +exp_scaling=general $SEEDS model.width_coefficient=1.0,1.5,2.0
#$PY_TEST -m +exp_scaling=general model.width_coefficient=2.5

# resolution
# 128 = 129 GFLOPs
# 160 = 202 GFLOPs
# 192 = 291 GFLOPs
# 224 = 397 GFLOPs
$PY_TEST -m +exp_scaling=general training.dataset.resolution=224


# depth
# $PY_SCRIPT -m +exp_scaling=general model.depth_coefficient=1.0,1.5