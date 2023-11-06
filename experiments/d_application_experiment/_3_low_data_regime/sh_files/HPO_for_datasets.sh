#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"
export PY_HPO="python experiments/d_application_experiment/_3_low_data_regime/HPO.py"


# 1
$PY_HPO dataset=isic2019 \
    HPOs.eq_nasnet.done=True \
    HPOs.eq_nasnet.done=True \
    HPOs.efficientnet.done=True HPOs.efficientnet_pre.done=True \
    HPOs.vit.done=True \
    debug=2
