# Get the data from ADNI1

conda activate clinicaEnv

# get into bids format


python src/data/mri/post_process_pipeline.py --run_pipeline True \
    --path "~/Data/frischs/datasets/adni/adni1" \
    --filter_type T3