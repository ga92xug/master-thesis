import os
import subprocess
from ax import Runner
import wandb

from util import encode_parameters


class HydraWandbRunner(Runner):
    def __init__(self, script_path, wandb_project, wandb_mode="disabled"):
        super().__init__()
        self.command_prefix = "python"
        self.script_path = script_path
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode

    def run(self, trial):
        # Construct the command
        command = [self.command_prefix, self.script_path]

        # Add the trial parameters
        trial_index = trial.index
        trial_params = trial.parameters
        encoded_params = encode_parameters(trial_params)
        command.extend([f"model=eq_nasnet"])
        command.extend([f"model.block_args={encoded_params}"])

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


    # This method returns a JSON-serializable representation of the runner
    def to_json(self):
        return {
            "script_path": self.script_path,
            "wandb_project": self.wandb_project,
            "wandb_mode": self.wandb_mode,
        }

    # This classmethod takes a JSON-serializable representation and returns a new runner
    @classmethod
    def from_json(cls, json_repr):
        return cls(
            script_path=json_repr["script_path"],
            wandb_project=json_repr["wandb_project"],
            wandb_mode=json_repr["wandb_mode"],
        )

    
