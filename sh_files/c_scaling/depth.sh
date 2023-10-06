#!/bin/bash

TAGS="wandb.tags=[constant_scheduler, depth2_scaling]"
SEEDS="other.seed=0,1,2"

# d1 is baseline 
# d2 is baseline 4 blocks

#$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=96 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}" model.depth_coefficient=2 $TAGS

#$PY_SCRIPT -m $SEEDS +exp_scaling=depth_scaling model.skip=conv model.depth_coefficient=1.5
#$PY_SCRIPT -m $SEEDS +exp_scaling=general model.depth_coefficient=1.5
#$PY_SCRIPT -m $EPOCHS +exp_scaling=depth_scaling model.skip=conv model.depth_coefficient=1.0

#$PY_SCRIPT -m $SEEDS +exp_scaling=general model.depth_coefficient=2 training=isic2019-training model.not_increase_1_layer=True

$PY_SCRIPT -m $SEEDS $TAGS +exp_scaling=general model.depth_coefficient=2 model.not_increase_1_layer=True model/blocks_args_dict=3blocks,4blocks