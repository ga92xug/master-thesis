export PY_SCRIPT="python src/main.py"
export PRE_TRAINED="train.network.pre_trained=logs/pretrain/2024-02-25_19-32-25/checkpoints/epoch_039.ckpt"
export SEEDS="seed=0,1,2,3,4"

scripts/4_application/pre_train_exp/ld_deepdrip_eq_nans_pre_no_schedule.sh
#scripts/4_application/pre_train_exp/ld_deepdrip_eq_nans_pre.sh
#scripts/4_application/pre_train_exp/ld_blood_eq_nans_pre.sh

$PY_SCRIPT hparams_search=ax_isic2019