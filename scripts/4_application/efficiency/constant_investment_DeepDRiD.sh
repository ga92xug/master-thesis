#!/bin/bash
DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"

SEEDS="other.seed=0,1,2,3,4"

# eq_nasnet
# HPs from efficientnet
python training/main.py -m +HPs_DeepDRiD=eq_nasnet \
    wandb.tags=[efficiency] \
    $SEEDS \
    training.max_gflops=9207.4604832 \
    training.earlystop.stop=False \
    training.eval_frequency=0.1 \
    model.dropout_rate=0.2 \
    training.optimizer.lr=0.0013960833048305068 \
    training.optimizer.weight_decay=0.00000265302216705901

# efficientnet
python training/main.py -m +HPs_DeepDRiD=efficientnet \
    wandb.tags=[efficiency] wandb.notes=fair_baseline_no_earlystop \
    $SEEDS \
    training.max_gflops=9207.4604832 \
    training.earlystop.stop=False 

# vit
python training/main.py -m +HPs_DeepDRiD=vit \
    wandb.tags=[efficiency] \
    $SEEDS \
    training.max_gflops=9207.4604832 \
    training.earlystop.stop=False \
    training.eval_frequency=0.1 \

