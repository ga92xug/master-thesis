#!/bin/bash

# Define the source directory where the runs are located
source_dir="/home/atuin/b180dc/b180dc27/logs/train/runs"

# Iterate through each subdirectory in the source directory
for run_folder in "$source_dir"/*; do
    # Check if the subdirectory is a directory
    if [ -d "$run_folder" ]; then
        # Find and store the offline-run directory (if it exists)
        offline_run_dir=$(find "$run_folder" -type d -name "offline-run-*" -print -quit)
        
        # Check if an offline-run directory was found
        if [ -n "$offline_run_dir" ]; then
            # Use wandb sync to upload the offline-run directory
            wandb sync "$offline_run_dir" --no-include-online
            
            # Print a message indicating that the upload is done
            echo "Uploaded $offline_run_dir"
        fi
    fi
done
