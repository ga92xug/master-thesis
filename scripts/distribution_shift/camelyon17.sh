#!/bin/bash
SEEDS="seed=0,1,2,3,4,5,6,7,8,9"
BASE_EXP="experiment=HPs/camelyon17/"
EXPERIMENTS=(
    "eq_nasnet_pre"
    "eq_nasnet"
    "efficientnet"
    "efficientnet_pre"
    "vit"
    "vit_pre"
)

for EXP in "${EXPERIMENTS[@]}"; do
    python src/main.py -m "${BASE_EXP}${EXP}" $SEEDS
done
