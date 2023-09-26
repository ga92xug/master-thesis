#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"

$PY_SCRIPT training=cifar10-training model=eq_wrn
python experiments/b_NAS/common_best_architecture/run_strategies.py datasets=[mnist_rot]

python PY_SCRIPT -m +exp_nas_ablation=se other.seed=0,1  model.blocks_args_dict._3.se_ratio=0.75
python PY_SCRIPT -m +exp_nas_ablation=convolutions model.width_coefficient=0.6
#python experiments/c_scaling/1_individual_scaling.py
