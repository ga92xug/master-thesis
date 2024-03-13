SEEDS="seed=0,1,2,3,4,5,6,7,8,9"
EXPERIMENTS=(
    "experiment=application/HPs_camelyon17/eq_nasnet_pre"
    "experiment=application/HPs_camelyon17/eq_nasnet"
    #"experiment=application/HPs_camelyon17/efficientnet"
    #"experiment=application/HPs_camelyon17/efficientnet_pre"
    #"experiment=application/HPs_camelyon17/vit"
    #"experiment=application/HPs_camelyon17/vit_pre"
)

for EXP in "${EXPERIMENTS[@]}"; do
    python src/main.py -m "$EXP" $SEEDS
done
