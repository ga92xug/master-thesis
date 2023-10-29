#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"
export PY_HPO="python experiments/d_application_experiment/_3_low_data_regime/low_data_HPO_new.py"

#$PY_HPO dataset=DeepDRiD low_data_regime=0.1

$PY_HPO dataset=Blood low_data_regime=1 \
    HPOs.eq_nasnet.done=True HPOs.vit.done=True HPOs.vit_pre.done=True
#$PY_HPO -m dataset=Blood low_data_regime=0.1,0.3
