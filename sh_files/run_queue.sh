#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"

python training/main.py -m \
    +exp_scaling=general wandb.tags=[constant_scheduler,depth2_scaling] \
    other.seed=2,3 \
    training.dataset.resolution=96 \
    model.width_coefficient=2 \
    model.depth_coefficient=1 model.not_increase_1_layer=True \
    model/blocks_args_dict=3blocks 
#$PY_SCRIPT -m other.seed=0,1,2 +exp_scaling=general training=isic2019-training-constant_scheduler model/blocks_args_dict=4blocks training.dataset.resolution=192 wandb.tags=[constant_scheduler]

#bash sh_files/d_application/baseline.sh
#bash sh_files/c_scaling/depth.sh
#bash sh_files/c_scaling/width.sh
#bash sh_files/c_scaling/resolution.sh
