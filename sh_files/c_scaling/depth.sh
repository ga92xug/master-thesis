#!/bin/bash

TAGS="wandb.tags=[depth2_scaling]"
SEEDS="other.seed=0,1"

# d1 is baseline 
# d2 is baseline 4 blocks

$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=96 "model.increase_blocks={2:{num_new_blocks:2,replace:{out_channel:1}}}" wandb.notes="5 blocks" $TAGS
$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=96 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}" model.depth_coefficient=1.4 wandb.notes="4 blocks depth 1.4" $TAGS

#$PY_SCRIPT -m $SEEDS +exp_scaling=depth_scaling model.skip=conv model.depth_coefficient=1.5
#$PY_SCRIPT -m $SEEDS +exp_scaling=general model.depth_coefficient=1.5
#$PY_SCRIPT -m $EPOCHS +exp_scaling=depth_scaling model.skip=conv model.depth_coefficient=1.0
