#!/bin/bash
PY_SCRIPT="python src/main.py"


$PY_SCRIPT -m train=isic2019 \
    train/network=wrn,efficientnet


$PY_SCRIPT -m train=isic2019 \
    train/network=eq_wrn \
    train.network.restrict=[none,none],[none,halved],[halved,halved] \
    train.network.drop_out=0.3 \