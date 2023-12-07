#!/bin/bash

SEEDS="other.seed=0,1,2"

# baseline
# 72 GFLOPs
#$PY_SCRIPT -m +exp_scaling=general $SEEDS wandb.notes="new_baseline"

# width_coefficient
# 1.5 = 149 GFLOPs
# 2.0 = 252 GFLOPs
# 2.5 = 381 GFLOPs
$PY_SCRIPT -m +exp_scaling=general $SEEDS model.width_coefficient=2.5
