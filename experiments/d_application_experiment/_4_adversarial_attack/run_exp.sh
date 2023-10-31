#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"
export PY_HPO="python experiments/d_application_experiment/_3_low_data_regime/low_data_HPO_new.py"

conda activate scaling-adversarial


#$PY_SCRIPT -m other.seed=0 +exp_HPO_blood=efficientnet_pre \
#    wandb.tags=[adversarial_attack,blood] other.should_test=True \
#    training.adversarial_attack=Foolbox \
#    other.verbose=2 other.debug=False other.backup_model=True \
#    training.epochs=2

# adversarial examples
$PY_SCRIPT -m other.seed=0 '+exp_HPO_blood=glob(*)' \
    wandb.tags=[adversarial_attack,blood] other.should_test=True \
    training.adversarial_attack=Foolbox \
    other.verbose=2 other.debug=False other.backup_model=True

# if results are bad try other attacks
