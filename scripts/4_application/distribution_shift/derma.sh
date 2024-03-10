export PY_SCRIPT="python src/main.py"
#export SEEDS="seed=0,1,2,3,4"


$PY_SCRIPT -m experiment=application/HPs_isic2019/efficientnet \
    tags="[dist_shift_derma]" task_name=derma \
    train=derma \

