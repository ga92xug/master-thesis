#!/bin/bash
export PY_HPO="python experiments/d_application_experiment/low_data_regime/HPO.py"

$PY_HPO dataset=Blood \
    debug=2
