#!/bin/bash

# 2 datasets

python training/main.py -m +exp_HPO_blood=eq_nasnet \
    wandb.tags=[CNN_augmentation] \
    training.max_gflops=46390.9675499172 \
    other.should_test=True \
    wandb.mode=disabled
    #other.seed=0,1,2 
    

#python training/main.py -m +exp_HPO_blood=efficientnet training=blood-training \
#    wandb.tags=[CNN_augmentation] \
#    +training.dataset.augment.train='{RandomHorizontalFlip: {p: 0.5}, Own_RandomRotation: {degrees: [0,90,180,270]}}' \
#    other.seed=0,1,2 \
#    other.should_test=True

