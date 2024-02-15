python src/main.py hparams_search=camelyon \
    train/network=efficientnet \
    train.network.pretrained=True \

python src/main.py hparams_search=camelyon \
    train/network=efficientnet \

python src/main.py hparams_search=camelyon \
    train/network=eq_nasnet \

python src/main.py hparams_search=camelyon \
    train/network=vit \

python src/main.py hparams_search=camelyon \
    train/network=vit \
    train.network.pretrained=True \