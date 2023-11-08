#!/bin/bash

# 2 datasets

# python training/main.py -m +exp_HPs_blood=eq_nasnet \
#     wandb.tags=[CNN_augmentation] \
#     other.seed=0,1 \
#     other.should_test=True \
#     wandb.mode=online \
#     training.max_gflops=60652.501743708 \
#     model.dropout_rate=0.5
    #training.epochs=4 \
    #training.optimizer.lr=0.0075,0.0025
    # 
    #training.scheduler=null \
    #+training.scheduler._target_=torch.optim.lr_scheduler.MultiStepLR \
    #+training.scheduler.milestones=[0,1] \
    #+training.scheduler.gamma=0.1
    #other.seed=0,1,2 
    
python training/main.py -m +exp_HPs_blood=efficientnet \
    wandb.tags=[CNN_augmentation] wandb.notes=fair_baseline_no_earlystop \
    other.seed=0,1,2 \
    other.should_test=True \
    training.epochs=37 \
    training.earlystop.stop=False 

