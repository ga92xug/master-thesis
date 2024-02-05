#!/bin/bash              
#     
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH -o /home/atuin/b180dc/b180dc27/logs/slurm/run.out
#SBATCH -e /home/atuin/b180dc/b180dc27/logs/slurm/run.err
#SBATCH --job-name=pre-training_imagenet
source activate scaling
echo "Virtual environment activated"
srun python src/main.py experiment=pre-training/imagenet local=alex \
    hardware.strategy=ddp hardware.devices=2 \
    training_setup.dataset.batch_size=100 \
    training_setup.trainer.max_epochs=1 training_setup.trainer.min_epochs=1  



#SBATCH --mem=400000
#SBATCH --nodes=1
#SBATCH --qos=a100multi
#SBATCH --ntasks-per-node=16