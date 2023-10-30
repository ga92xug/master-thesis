#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"
export PY_HPO="python experiments/d_application_experiment/_3_low_data_regime/low_data_HPO_new.py"


#$PY_HPO dataset=Blood low_data_regime=1 \
#    HPOs.eq_nasnet.done=True HPOs.vit.done=True HPOs.vit_pre.done=True
#$PY_HPO -m dataset=Blood low_data_regime=0.1,0.3

# 0.1
$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPO_blood=efficientnet,efficientnet_pre \
    wandb.tags=[Blood_low_data] other.should_test=True \
    training.dataset.reduction_factor=0.1 \
    training.epochs=300 training.earlystop.patience=50

# 0.3
$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPO_blood=efficientnet,efficientnet_pre \
    wandb.tags=[Blood_low_data] other.should_test=True \
    training.dataset.reduction_factor=0.3 \
    training.epochs=200 training.earlystop.patience=40


# 0.5
$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPO_blood=efficientnet,efficientnet_pre \
    wandb.tags=[Blood_low_data] other.should_test=True \
    training.dataset.reduction_factor=0.5 \
    training.epochs=150 training.earlystop.patience=30

# 1
# here we only have to run eq_nasnet
#$PY_SCRIPT -m other.seed=0,1,2,3,4 +exp_HPO_blood=eq_nasnet \
#    wandb.tags=[Blood_low_data] other.should_test=True \
#    training.dataset.reduction_factor=1


# adversarial examples
#$PY_SCRIPT -m other.seed=0,1 '+exp_HPO_blood=glob(*)' \
#    wandb.tags=[adversarial_attack,blood] other.should_test=True \
#    training.adversarial_attack=AutoAttack
# if results are bad try other attacks


# increase rotation for images in efficientnet, vit
    