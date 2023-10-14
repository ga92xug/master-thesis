#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"

# efficientnet pre F
wandb agent ga92xug/SL-sweeps/xqym1gv7
# efficientnet pre T
wandb agent ga92xug/SL-sweeps/nofqluq8
# vit pre F
wandb agent ga92xug/SL-sweeps/3bwa984b
# vit pre T
wandb agent ga92xug/SL-sweeps/xp8pz6zv

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


