import os
import subprocess
import time
from ax import Runner
import wandb
from omegaconf import OmegaConf

from networks.eq_nasnet.util import encode_parameters

class HydraWandbRunner(Runner):
    def __init__(
            self, 
            script_path: str, 
            wandb_entity: str, 
            wandb_project: str, 
            wandb_mode: str, 
            choice_2_range_param: dict, 
            strides: list, 
            training_dict: dict,
            verbose: int = 0,
        ):
        """
        Args:
            script_path: Path to the script to run.
            wandb_entity: The entity to use for wandb.
            wandb_project: The project to use for wandb.
            wandb_mode: The mode to use for wandb.
            choice_2_range_param: A dict mapping choice parameters to their
                corresponding range parameters.
            strides: A list of strides to use for the search space.
            training_dict: A dict of training parameters.
        """
        super().__init__()
        self.command_prefix = "python"
        self.script_path = script_path
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.choice_2_range_param = choice_2_range_param
        self.strides = strides
        self.training_dict = training_dict
        self.verbose = verbose

    def run(self, trial):
        trial_params, trial_index = trial
        # Construct the command
        command = [self.command_prefix, self.script_path]

        # Add the trial parameters
        #trial_index = trial.index
        #print(trial)
        #arm = trial.arm
        #trial_params = arm.parameters

        # Create a new wandb run
        wandb_run = wandb.init(
            entity=self.wandb_entity,
            project=self.wandb_project, 
            mode=self.wandb_mode,
            name=str(trial_index),
        )

        # encode the search space parameters
        assert False, "currently not working. Logic of encode params changed"
        encoded_params = encode_parameters(trial_params, 
                                           self.choice_2_range_param, 
                                           self.strides)
        if self.verbose >= 1: 
            print("trial_params", trial_params)
            print("encoded_params", encoded_params)

        # pass the encoded search space parameter
        command.extend([f"model.blocks_args={encoded_params}"])
        
        # Iterate over key-value pairs in the 'training' dict
        for key, value in self.training_dict.items():
            command.append(f"{key}={value}")
                
        # pass wandb parameters
        command.extend([f"wandb.entity={self.wandb_entity}",
                        f"wandb.project={self.wandb_project}",
                        f"wandb.run_id={wandb_run.id}",
                        f"wandb.mode={self.wandb_mode}"])

        # run the training
        subprocess.run(command, stdout=subprocess.DEVNULL if self.verbose <= 0 else None)
        

        # Return the trial metadata
        trial_metadata = {
            "name": str(trial_index),
            "wandb_run_id": wandb_run.id,
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

    
