#!/bin/bash
DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"

SEEDS="other.seed=0,1,2,3,4"

# eq_nasnet
# HPs from efficientnet
python training/main.py -m +HPs_blood=eq_nasnet \
    wandb.tags=[efficiency] \
    $SEEDS \
    training.max_gflops=45898.231398285 \
    training.earlystop.stop=False \
    training.eval_frequency=0.1 \
    model.dropout_rate=0.2 \
    training.optimizer.lr=0.0005258641063603519 \
    training.optimizer.weight_decay=0.00000165943876390034
   
# efficientnet
python training/main.py -m +HPs_blood=efficientnet \
    wandb.tags=[efficiency] wandb.notes=fair_baseline_no_earlystop \
    $SEEDS \
    training.max_gflops=45898.231398285 \
    training.earlystop.stop=False 

# vit
python training/main.py -m +HPs_blood=vit \
    wandb.tags=[efficiency] \
    $SEEDS \
    training.max_gflops=45898.231398285 \
    training.earlystop.stop=False \
    training.eval_frequency=0.1 \

