PRE_TRAINED="train.network.pre_trained=logs/pretrain/2024-02-25_19-32-25/checkpoints/epoch_039.ckpt"

python src/main.py -m experiment=application/HPs_camelyon17/eq_nasnet \
    $PRE_TRAINED

