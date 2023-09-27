#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"

# NAS ablation 
# se
#$PY_SCRIPT -m +exp_nas_ablation=se other.seed=0,1,2  model.blocks_args_dict._3.se_ratio=0.0
$PY_SCRIPT -m +exp_nas_ablation=se other.seed=0,1,2  model.blocks_args_dict._3.se_ratio=0.75
# conv
$PY_SCRIPT -m +exp_nas_ablation=convolutions model.width_coefficient=0.6 model.conv_op=conv
$PY_SCRIPT -m +exp_nas_ablation=convolutions model.width_coefficient=1.1 model.conv_op=dconv
# kernel size
$PY_SCRIPT -m +exp_nas_ablation=kernel_size