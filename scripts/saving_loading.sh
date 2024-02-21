# saving
#python src/main.py train=imagenette \
#    seed=0 \
#    task_name=pretrained_test \
#    train.trainer.max_epochs=20 \
#    train.callbacks.model_checkpoint.save_top_k=-1

#exit 0
# loading
python src/main.py train=imagenette \
    train_mode=train_continue \
    ckpt_path=/home/frischs/dev1/scaling-laws-ecnn/logs/pretrained_test/runs/2024-02-21_16-40-30/checkpoints/epoch_003.ckpt \
    task_name=pretrained_test \
    seed=0