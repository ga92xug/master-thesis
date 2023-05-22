import os
import subprocess
from ax import Runner
import wandb

class HydraWandbRunner(Runner):
    def __init__(self, script_path, wandb_project, wandb_mode="disabled"):
        super().__init__()
        self.command_prefix = "python"
        self.script_path = script_path
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode

    def run(self, trial):
        trial_index = trial.index
        trial_params = trial.parameters

        # Construct the command
        command = [self.command_prefix, self.script_path]
        for key, value in trial_params.items():
            command.extend([f"{key}={value}"])

        # Create a new wandb run
        run = wandb.init(project=self.wandb_project, mode=trial_params["wandb.mode"])
        run_id = run.id
        
        # Run the command
        subprocess.run(command)

        # Return the trial metadata
        trial_metadata = {
            "name": str(trial_index),
            "wandb_run_id": run_id,
        }

        return trial_metadata