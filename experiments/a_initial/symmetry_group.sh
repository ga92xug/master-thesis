#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"


# initial group increase
$PY_SCRIPT -m +experiments=initial \
    rotation=2,4,8,12 \
    wandb.tags=[initial_group_increase]

# restriction
$PY_SCRIPT -m +experiments=initial \
    model.restrict=[invariant,invariant][halved,invariant],[halved,halved],[none,none] \
    training=mnist_rot-training,ciar10-training,galaxy10-training \
    wandb.tags=[restriction]

# cyclic or dihedral
$PY_SCRIPT -m +experiments=initial \
    model.group=cyclic,dihedral \
    model.depth=16,22 \
    model.width_coefficient=4,6 \
    wandb.tags=[cyclic_or_dihedral]

# interpolation artifacts

# no adjust
$PY_SCRIPT -m +experiments=initial \
    model.kernel_layout=[3,3],[5,5] \
    model.rotation=2,4,6,8,10,12 \
    wandb.tags=[interpolation_artifacts_no_adjust]

# adjust
# w=2.5 try and error found
$PY_SCRIPT -m +experiments=initial \
    model.kernel_layout=[5,5] \
    model.width_coefficient=2.5 \
    model.rotation=2,4,6,8,10,12 \
    wandb.tags=[interpolation_artifacts_adjust]