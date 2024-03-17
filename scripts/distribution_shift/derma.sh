DERMA="tags="[dist_shift_derma]" task_name=domain_shift train=derma"
SEEDS="seed=0,1,2,3,4"
BASE_EXP="experiment=HPs/isic2019/" 
# the HPs should be similar since isic2019 is to a large extend HAM10000
EXPERIMENTS=(
    "eq_nasnet_pre"
    "eq_nasnet"
    "efficientnet"
    "efficientnet_pre"
    "vit"
    "vit_pre"
)

for EXP in "${EXPERIMENTS[@]}"; do
    python src/main.py -m "${BASE_EXP}${EXP}" $SEEDS $DERMA
done