#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"

#$PY_SCRIPT -m +exp_nas_ablation=se other.seed=0  model.blocks_args_dict._3.se_ratio=0.0
bash sh_files/b_NAS/other_datasets.sh
bash sh_files/c_scaling/depth_scaling.sh