# saving
python src/main.py train=imagenette \
    task_name=pretrained_test \
#    #train.trainer.max_epochs=5 \
#
#exit 0
# loading
#python src/main.py train=imagenette \
#    train_mode=train_continue \
#    ckpt_path=/home/frischs/dev1/scaling-laws-ecnn/logs/pretrained_test/runs/2024-02-19_22-10-12/checkpoints/epoch_002.ckpt \
#    task_name=pretrained_test \
#    seed=41