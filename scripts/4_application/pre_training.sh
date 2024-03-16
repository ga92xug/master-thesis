python src/main.py experiment=pre-training/imagenet_100 test_mode=no_test \
    task_name=pretrain \
    train.dataset.batch_size=128 \
    train.dataset.workers=8 \
    tags='["decrease_pooling_size","lastgroupD1"]' \
    train.network.blocks_args_dict._4.reflection=0 \
    train.network.blocks_args_dict._4.group=1 \
    train_mode=train_continue \
    ckpt_path=logs/pretrain/runs/2024-03-05_15-15-31/checkpoints/epoch_042.ckpt \
    train.trainer.accumulate_grad_batches=8 \