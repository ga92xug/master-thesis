#!/bin/bash
DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"


$PY_SCRIPT training=isic2019-training model=eq_wrn
$PY_SCRIPT training=isic2019-training model=wrn
$PY_SCRIPT training=isic2019-training model=densenet