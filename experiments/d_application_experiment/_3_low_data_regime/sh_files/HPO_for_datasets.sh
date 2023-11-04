#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"
export PY_HPO="python experiments/d_application_experiment/_3_low_data_regime/low_data_HPO_new.py"


# 1
$PY_HPO dataset=isic2019 \
    debug=0
