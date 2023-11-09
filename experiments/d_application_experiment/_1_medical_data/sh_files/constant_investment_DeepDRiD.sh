#!/bin/bash


#9207.4604832

python training/main.py -m +exp_HPs_DeepDRiD=eq_nasnet \
    training.earlystop.stop=False \
    wandb.tags=[CNN_augmentation_DeepDRiD] \
    other.seed=0,1 \
    other.should_test=True \
    wandb.mode=online \
    training.max_gflops=9207.4604832 \
    model.dropout_rate=0.5
    #training.epochs=4 \
    #training.optimizer.lr=0.0075,0.0025
    #training.scheduler=null \
    #+training.scheduler._target_=torch.optim.lr_scheduler.MultiStepLR \
    #+training.scheduler.milestones=[0,1] \
    #+training.scheduler.gamma=0.1
    #other.seed=0,1,2 
    
python training/main.py -m +exp_HPs_DeepDRiD=efficientnet \
    wandb.tags=[CNN_augmentation_DeepDRiD] wandb.notes=fair_baseline_no_earlystop \
    other.seed=0,1 \
    other.should_test=True \
    training.max_gflops=9207.4604832 \
    training.earlystop.stop=False 

