#!/bin/bash

python src/main.py hparams_search=ax_oct_slurm train/network=efficientnet
python src/main.py hparams_search=ax_oct_slurm train/network=efficientnet_pre 
