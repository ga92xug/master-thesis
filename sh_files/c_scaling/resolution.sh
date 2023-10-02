#!/bin/bash
SEEDS="other.seed=0,1,2"


# resolution
# 128 = 129 GFLOPs
# 160 = 202 GFLOPs
# 192 = 291 GFLOPs
# 224 = 397 GFLOPs
# training.dataset.resolution=128,192
#$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128,192 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}"
#$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128,192 model.depth_coefficient=1.5

# 4 blocks baseline
$PY_SCRIPT -m other.seed=1,2 +exp_scaling=general training.dataset.resolution=96 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}" wandb.notes="4 blocks baseline" wandb.tags="[resolution_scaling_baseline, resolution_scaling]"
$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=128,196,224 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}" wandb.tags="[resolution_scaling]"

# increase with 1+1 block or 1 block + depthwise
#$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=192 "model.increase_blocks={2:{num_new_blocks:2,replace:{out_channel:1}}}" wandb.notes="increase with 1+1 block or 1 block + depthwise, resolution_192"
#$PY_SCRIPT -m $SEEDS +exp_scaling=general training.dataset.resolution=192 "model.increase_blocks={2:{num_new_blocks:1,replace:{out_channel:1}}}" model.depth_coefficient=1.4 wandb.notes="increase with 1+1 block or 1 block + depthwise, resolution_192"


# idea for run queue
# always only start 1 job -> 5-6 next jobs in different files -> modify on the go