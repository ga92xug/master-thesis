#!/bin/bash

# best epoch 35

python training/main.py -m +exp_HPs_isic2019=efficientnet \
    wandb.tags=[CNN_augmentation_isic2019] wandb.notes=fixed_compute \
    other.seed=2,3,4 \
    other.should_test=True \
    training.earlystop.stop=False \
    training.max_gflops=89955.244731492 \

    # training.earlystop.monitor=valid.acc_weighted \
    # training.earlystop.mode=max \
    # training.earlystop.patience=20 \


python training/main.py -m +exp_HPs_isic2019=eq_nasnet \
    training.earlystop.stop=False \
    wandb.tags=[CNN_augmentation_isic2019] wandb.notes=0.2_dropout \
    other.seed=0,1,2,3,4 \
    other.should_test=True \
    training.max_gflops=89955.244731492 \
    model.dropout_rate=0.2

    

