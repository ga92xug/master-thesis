SEEDS="seed=0,1,2,3,4"

python src/main.py -m training_setup=b_camelyon17-training \
    $SEEDS \
    training_setup/network=eq_nasnet \
    training_setup.dataset.batch_size=512

python src/main.py -m training_setup=b_camelyon17-training \
    $SEEDS \
    training_setup/network=efficientnet \
    training_setup.dataset.batch_size=2048
    
python src/main.py -m training_setup=b_camelyon17-training \
    $SEEDS \
    training_setup/network=vit \
    training_setup.dataset.batch_size=256