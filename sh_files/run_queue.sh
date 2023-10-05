#!/bin/bash

export DEBUG_MODE="other.debug=True"
export PY_SCRIPT="python training/main.py"
export PY_TEST="python training/model_instantiate.py"

#bash sh_files/d_application/baseline.sh
bash sh_files/c_scaling/compound.sh
#bash sh_files/c_scaling/resolution.sh