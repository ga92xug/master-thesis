#!/bin/bash

# best epoch 35

# python training/main.py -m +exp_HPs_isic2019=efficientnet \
#     wandb.tags=[CNN_augmentation_isic2019] wandb.notes=fixed_compute \
#     other.seed=0,1 \
#     other.should_test=True \
#     training.earlystop.stop=False \
#     training.max_gflops=89955.244731492 \
    # training.earlystop.monitor=valid.acc_weighted \
    # training.earlystop.mode=max \
    # training.earlystop.patience=20 \



python training/main.py -m +exp_HPs_isic2019=eq_nasnet \
    training.earlystop.stop=False \
    wandb.tags=[CNN_augmentation_isic2019] \
    other.seed=0,1 \
    other.should_test=True \
    wandb.mode=online \
    training.max_gflops=89955.244731492 \
    model.dropout_rate=0.5
    #training.epochs=4 \
    #training.optimizer.lr=0.0075,0.0025
    #training.scheduler=null \
    #+training.scheduler._target_=torch.optim.lr_scheduler.MultiStepLR \
    #+training.scheduler.milestones=[0,1] \
    #+training.scheduler.gamma=0.1
    #other.seed=0,1,2 
    
    

