#!/bin/bash

# This script takes a very long time to finish if not done in parallel.
# It is recommended to use a cluster with a job scheduler to run this script.
SLURM="hydra/launcher=slurm hydra.launcher.max_num_timeout=3 hydra.launcher.timeout_min=360 hydra.sweeper.max_batch_size=5"

# Define base strings for the hyperparameters search and network training
BASE_HP_SEARCH="hparams_search="
BASE_NETWORK="train/network="

# Array of datasets
DATASETS=(
    "oct"
    "isic2019"
    "blood"
    "camelyon17"
)

# Array of network configurations
NETWORKS=(
    "eq_nasnet_pre"
    "eq_nasnet"
    "efficientnet"
    "efficientnet_pre"
    "vit"
    "vit_pre"
)

# Iterate over datasets
for dataset in "${DATASETS[@]}"; do
    # Iterate over networks
    for network in "${NETWORKS[@]}"; do
        # Execute the python script with combined arguments
        python src/main.py "${BASE_HP_SEARCH}${dataset}" "${BASE_NETWORK}${network}" # $SLURM
    done
done
