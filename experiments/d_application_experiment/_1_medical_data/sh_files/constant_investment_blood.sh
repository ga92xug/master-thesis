#!/bin/bash

# 2 datasets

python training/main.py -m +exp_HPs_blood=eq_nasnet \
    wandb.tags=[CNN_augmentation_blood] \
    other.seed=0,1,2,3,4 \
    other.should_test=True \
    wandb.mode=online \
    training.max_gflops=57373.98813594 \
    model.dropout_rate=0.2 \
    training.earlystop.stop=False 
    
python training/main.py -m +exp_HPs_blood=efficientnet \
    wandb.tags=[CNN_augmentation_blood] wandb.notes=fair_baseline_no_earlystop \
    other.seed=0,1,2,3,4 \
    other.should_test=True \
    training.max_gflops=57373.98813594 \
    training.earlystop.stop=False 

