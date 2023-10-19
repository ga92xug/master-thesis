#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"

$PY_SCRIPT -m other.seed=0,1,2 '+exp_model_specific_overwrites=eq_nasnet_DeepDRiD' wandb.tags=[test_performance_HPO] other.should_test=True

# $PY_SCRIPT -m other.seed=0,1,2 training=DeepDRiD-training \
#     wandb.project=SL-Application \
#     model=eq_nasnet \
#     model/blocks_args_dict=4blocks \
#     other.should_test=True
# 
# $PY_SCRIPT -m other.seed=0,1,2 training=DeepDRiD-training \
#     wandb.project=SL-Application \
#     model=efficientnet model.pretrained=False,True \
#     other.should_test=True

# $PY_SCRIPT -m other.seed=0,1,2 training=DeepDRiD-training \
#     model=vit model.pretrained=False,True \
#     other.should_test=Tr
#     training.dataset.resolution=224


