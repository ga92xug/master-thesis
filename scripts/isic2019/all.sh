SEEDS="seed=0,1,2,3,4"
BASE_EXP="experiment=HPs/isic2019/"
EXPERIMENTS=(
    "eq_nasnet_pre"
    "eq_nasnet"
    "efficientnet"
    "efficientnet_pre"
    "vit"
    "vit_pre"
)

for EXP in "${EXPERIMENTS[@]}"; do
    python src/main.py -m "${BASE_EXP}${EXP}" $SEEDS
done
