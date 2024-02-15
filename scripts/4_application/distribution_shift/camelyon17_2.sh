python src/main.py experiment=application/HPs_camelyon17/efficientnet_pre

python src/main.py experiment=application/HPs_camelyon17/efficientnet

exit 0
  
python src/main.py hparams_search=camelyon \
    train/network=vit \

python src/main.py hparams_search=camelyon \
    train/network=vit \
    train.network.pretrained=True \