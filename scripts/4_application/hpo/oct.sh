#!/bin/bash

python src/main.py hparams_search=ax_oct_slurm train/network=efficientnet
python src/main.py hparams_search=ax_oct_slurm train/network=efficientnet_pre 

# done 
# tmux 1 -> vit
# tmux 2 -> eq_nasnet
# tmux 3 -> vit_pre

# missing
# vit_pre
# eq_nasnet_pre