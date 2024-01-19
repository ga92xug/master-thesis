#!/bin/bash
#SBATCH -N 1
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH -t 3-23:59:30
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=400000
#SBATCH -o /dss/dsshome1/lxc0A/ga96sac2/logs/run.out
#SBATCH -e /dss/dsshome1/lxc0A/ga96sac2/logs/run.err
source /dss/dsshome1/lxc0A/ga96sac2/envs/isometry/bin/activate
echo "Virtual environment activated"
srun torchrun --standalone --nnodes=1 --nproc_per_node=4 /dss/dsshome1/lxc0A/ga96sac2/scaling-laws/train.py \
    wandb.project=test wandb.run=test setup=test


salloc.tinygpu  --partition=a100 --gres=gpu:a100:1 --time=02:30:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 


#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
echo "Starting sbatch script"
srun ./cuda_application