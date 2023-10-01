#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
SEEDS="other.seed=0,1"
EPOCHS="training.epochs=100"


$PY_SCRIPT -m $EPOCHS $SEEDS +exp_scaling=depth_scaling model.skip=identity model.depth_coefficient=1.5
$PY_SCRIPT -m $EPOCHS $SEEDS +exp_scaling=depth_scaling model.skip=conv model.depth_coefficient=1.5
$PY_SCRIPT -m $EPOCHS $SEEDS +exp_scaling=general model.depth_coefficient=1.5
#$PY_SCRIPT -m $EPOCHS +exp_scaling=depth_scaling model.skip=conv model.depth_coefficient=1.0
