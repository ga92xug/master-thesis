#!/bin/bash

SEEDS="other.seed=0,1"

$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}" model.width_coefficient=1.5,2.0 wandb.notes="b4,r128,w1.5,2.0" wandb.tags="[compound_scaling]"