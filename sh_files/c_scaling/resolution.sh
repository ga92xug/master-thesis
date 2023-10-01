#!/bin/bash
SEEDS="other.seed=0"


# resolution
# 128 = 129 GFLOPs
# 160 = 202 GFLOPs
# 192 = 291 GFLOPs
# 224 = 397 GFLOPs
$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128,192 "model.increase_blocks={3:1}"
$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128,192 model.depth_coefficient=1.5
