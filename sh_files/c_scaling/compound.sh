#!/bin/bash

SEEDS="other.seed=1,2"

$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}" model.width_coefficient=1.5 model.depth_coefficient=1.0,1.4 wandb.tags="[compound_scaling]"
