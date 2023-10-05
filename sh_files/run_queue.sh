#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"

$PY_SCRIPT -m other.seed=1 +exp_scaling=general training.dataset.resolution=192 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}" wandb.tags="[resolution_scaling]"

#bash sh_files/d_application/baseline.sh
bash sh_files/c_scaling/compound.sh
#bash sh_files/c_scaling/resolution.sh