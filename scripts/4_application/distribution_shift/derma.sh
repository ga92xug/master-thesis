export PY_SCRIPT="python src/main.py"
export DERMA="tags="[dist_shift_derma]" task_name=derma train=derma"
#export SEEDS="seed=0,1,2,3,4"


#$PY_SCRIPT -m experiment=application/HPs_isic2019/efficientnet_pre \
#    $DERMA

$PY_SCRIPT -m experiment=application/HPs_isic2019/efficientnet \
    $DERMA
