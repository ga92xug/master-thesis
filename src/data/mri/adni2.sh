# Get the data from ADNI1

conda activate clinicaEnv

# get into bids format
clinica convert adni-to-bids [OPTIONS] DATASET_DIRECTORY CLINICAL_DATA_DIRECTORY BIDS_DIRECTORY


python src/data/mri/post_process_pipeline.py --run_pipeline True \
    --path "~/Data/frischs/datasets/adni/adni2" \
    --filter_type T1.5