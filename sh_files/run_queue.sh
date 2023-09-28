#!/bin/bash

DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"
PY_TEST="python training/model_instantiate.py"

bash sh_files/c_scaling/depth_scaling.sh
bash sh_files/c_scaling/width.sh