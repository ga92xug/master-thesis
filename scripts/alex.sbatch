#!/bin/bash              
#     
#SBATCH -N 1
#SBATCH -t 23:59:50
#SBATCH --partition=a40
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH -o /home/atuin/b180dc/b180dc27/logs/slurm/run.out
#SBATCH -e /home/atuin/b180dc/b180dc27/logs/slurm/run.err
#SBATCH --job-name=pre-training_imagenet
source activate scaling
echo "Virtual environment activated"
srun python src/main.py experiment=pre-training/imagenet local=alex

#SBATCH --mem=400000