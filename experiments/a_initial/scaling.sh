#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"


# width 
$PY_SCRIPT -m +experiments=initial \
    model.widen_factor=1,2,3,4 \

# depth
$PY_SCRIPT -m +experiments=initial \
    model.widen_factor=2 \
    model.depth=16,22,28,34,40 \