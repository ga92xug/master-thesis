SEEDS="seed=0,1,2,3,4,5,6,7,8,9"

python src/main.py -m experiment=application/HPs_camelyon17/eq_nasnet \
    $SEEDS \
    
python src/main.py -m 'experiment=application/HPs_camelyon17/efficientnet' \
    $SEEDS \

python src/main.py -m 'experiment=application/HPs_camelyon17/efficientnet_pre' \
    $SEEDS \

python src/main.py -m 'experiment=application/HPs_camelyon17/vit' \
    $SEEDS \

python src/main.py -m 'experiment=application/HPs_camelyon17/vit_pre' \
    $SEEDS \