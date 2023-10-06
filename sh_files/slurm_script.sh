#!/bin/bash
#SBATCH -N 1
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH -t 3-23:59:30
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=100000
#SBATCH -o /dss/dsshome1/lxc0A/ga96sac2/stefan/logs/run.out
#SBATCH -e /dss/dsshome1/lxc0A/ga96sac2/stefan/logs/run.err
source /dss/dsshome1/lxc0A/ga96sac2/envs/isometry/bin/activate
echo "Virtual environment activated"
srun torchrun --standalone --nnodes=1 --nproc_per_node=1 /dss/dsshome1/lxc0A/ga96sac2/scaling-laws/train.py \
    wandb.project=test wandb.run=test setup=test