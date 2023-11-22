#!/bin/bash


python training/main.py -m +exp_HPs_DeepDRiD=eq_nasnet \
    training.earlystop.stop=False \
    wandb.tags=[CNN_augmentation_DeepDRiD] \
    other.seed=0,1,2,3,4 \
    other.should_test=True \
    wandb.mode=online \
    training.max_gflops=9207.4604832 \
    model.dropout_rate=0.2 \
    training.earlystop.stop=False \
    training.eval_frequency=0.1 \
    
exit 0


#9207.4604832
python training/main.py -m +exp_HPs_DeepDRiD=efficientnet \
    wandb.tags=[CNN_augmentation_DeepDRiD] wandb.notes=fair_baseline_no_earlystop \
    other.seed=2,3,4 \
    other.should_test=True \
    training.max_gflops=9207.4604832 \
    training.earlystop.stop=False 