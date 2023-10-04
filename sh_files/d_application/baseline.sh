#!/bin/bash

SEEDS="other.seed=0,1,2"
TAGS=wandb.tags=[baseline_camelyon17]

# baseline
$PY_SCRIPT -m +exp_application=general $TAGS