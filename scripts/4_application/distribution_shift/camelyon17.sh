SEEDS="seed=0,1,2,3,4"

python src/main.py -m train=b_camelyon17-training \
    $SEEDS \
    train/network=eq_nasnet \
    train.dataset.batch_size=512

python src/main.py -m train=b_camelyon17-training \
    $SEEDS \
    train/network=efficientnet \
    train.dataset.batch_size=2048
    
python src/main.py -m train=b_camelyon17-training \
    $SEEDS \
    train/network=vit \
    train.dataset.batch_size=256