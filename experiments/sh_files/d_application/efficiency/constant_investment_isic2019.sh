#!/bin/bash


python training/main.py -m +exp_HPs_isic2019=eq_nasnet_efficiency \
    training.earlystop.stop=False \
    wandb.tags=[CNN_augmentation_isic2019] \
    other.seed=0,1,2,3,4 \
    other.should_test=True \
    training.max_gflops=107446.542318171 \
    model.dropout_rate=0.5 \
    #training.eval_frequency=0.1 \

# python training/main.py -m +exp_HPs_isic2019=efficientnet \
#     wandb.tags=[CNN_augmentation_isic2019] wandb.notes=fixed_compute \
#     other.seed=0,1,2,3,4 \
#     other.should_test=True \
#     training.earlystop.stop=False \
#     training.max_gflops=107446.542318171 \

# best epoch 43

# python training/main.py -m +exp_HPs_isic2019=efficientnet \
#     wandb.tags=[CNN_augmentation_isic2019] \
#     other.seed=0,1,2,3,4 \
#     other.should_test=True \
#     training.earlystop.stop=True \
#     training.earlystop.monitor=valid.acc_weighted \
#     training.earlystop.mode=max \