#!/bin/bash

# 2 datasets

python training/main.py -m +exp_HPs_DeepDRiD=eq_nasnet \
    wandb.tags=[CNN_augmentation] \
    other.seed=0,1 \
    other.should_test=True \
    wandb.mode=online \
    training.max_gflops=6576.757488 \
    #training.epochs=4 \
    #training.optimizer.lr=0.0075,0.0025
    # training.max_gflops=46390.9675499172 \
    #training.scheduler=null \
    #+training.scheduler._target_=torch.optim.lr_scheduler.MultiStepLR \
    #+training.scheduler.milestones=[0,1] \
    #+training.scheduler.gamma=0.1
    #other.seed=0,1,2 
    
#python training/main.py -m +exp_HPs_DeepDRiD=efficientnet \
#    wandb.tags=[CNN_augmentation]  \
#    +training.dataset.augment.train.RandomHorizontalFlip='{p: 0.5}' \
#    +training.dataset.augment.train.Own_RandomRotation='{degrees: [0,90,180,270]}' \
#    other.seed=0,1,2 \
#    other.should_test=True \
    #training.earlystop.stop=False 
    #training.epochs=28 \

# experiments/d_application_experiment/_1_medical_data/increase_dataset.sh