import sys
import os
sys.path.append(os.getcwd())
from scripts.low_data.build_low_data_command import build_slurm_command_low_data

def blood():
    experiment_location = "experiment=HPs/blood/"
    args = [
    # reduction_factor, max_epochs, patience, val_check
        (0.05, 200, 200, 10),
        (0.1, 150, 150, 5),
        (0.3, 125, 125, 3),
        (0.5, 100, 100, 2),
        (1, 50, 50, 1),
    ]
    build_slurm_command_low_data(
        experiment_location, args, execute=False, local=True,
        submitit_logs="logs/submitit_logs/"
    )

def oct():
    experiment_location = "experiment=HPs/oct/"
    args = [
    # reduction_factor, max_epochs, patience, val_check
        (0.05, 50, 25, 5),
        (0.1, 40, 20, 4),
        (0.3, 30, 15, 2),
        (0.5, 20, 10, 2),
        (1, 10, 5, 1),
    ]
    build_slurm_command_low_data(
        experiment_location, args, execute=False, local=True,
        submitit_logs="logs/submitit_logs/"
    )

if __name__ == "__main__":
    #blood()
    oct()