#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"


#$PY_SCRIPT -m other.seed=0,1,2 +exp_scaling=general training=isic2019-training-constant_scheduler model/blocks_args_dict=4blocks training.dataset.resolution=192 wandb.tags=[constant_scheduler]

#bash sh_files/d_application/baseline.sh
bash sh_files/c_scaling/depth.sh
#bash sh_files/c_scaling/width.sh
#bash sh_files/c_scaling/resolution.sh
