#!/bin/bash

python training/main.py -m +exp_HPs_blood=vit \
    wandb.tags=[CNN_augmentation_blood] \
    other.seed=0,1,2,3,4 \
    other.should_test=True \
    training.max_gflops=45898.231398285 \
    training.earlystop.stop=False \
    training.eval_frequency=0.1 \

exit 0

#python training/main.py -m +exp_HPs_blood=eq_nasnet \
#    wandb.tags=[CNN_augmentation_blood] \
#    other.seed=0,1,2,3,4 \
#    other.should_test=True \
#    training.max_gflops=45898.231398285 \
#    model.dropout_rate=0.2 \
#    training.earlystop.stop=False \
#    training.eval_frequency=0.1 \
    
# python training/main.py -m +exp_HPs_blood=efficientnet \
#     wandb.tags=[CNN_augmentation_blood] wandb.notes=fair_baseline_no_earlystop \
#     other.seed=0,1,2,3,4 \
#     other.should_test=True \
#     training.max_gflops=45898.231398285 \
#     training.earlystop.stop=False 

