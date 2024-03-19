#!/bin/bash
python src/main.py experiment=pre-training/imagenet_100 test_mode=no_test \
    train.dataset.batch_size=128 \
    train.dataset.workers=8 \
    train.trainer.accumulate_grad_batches=8 \
    #train_mode=train_continue \
    #ckpt_path=logs/pretrain/runs/2024-03-05_15-15-31/checkpoints/epoch_042.ckpt \
    