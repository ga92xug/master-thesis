#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"


# initial group increase
$PY_SCRIPT +experiments=initial \
    model.rotation=2 \
    wandb.tags=[initial_group_increase] \
    $DEBUG_MODE