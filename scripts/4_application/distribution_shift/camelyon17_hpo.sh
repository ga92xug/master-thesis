python src/main.py hparams_search=camelyon \
    train/network=efficientnet \
    train.network.pretrained=True \

python src/main.py hparams_search=camelyon \
    train/network=efficientnet \

exit 0
  
python src/main.py hparams_search=camelyon \
    train/network=vit \

python src/main.py hparams_search=camelyon \
    train/network=vit \
    train.network.pretrained=True \