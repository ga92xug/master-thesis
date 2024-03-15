#!/bin/bash
DEBUG_MODE="other.debug=True"
PY_SCRIPT="python training/main.py"


#$PY_SCRIPT training=isic2019-training model=eq_wrn
$PY_SCRIPT -m training=isic2019-training \
    model=wrn,efficientnet


$PY_SCRIPT -m training=isic2019-training \
    model=eq_wrn model.restrict=[none,none],[none,halved],[halved,halved] \
    model.drop_out=0.3 \
    other.debug=True