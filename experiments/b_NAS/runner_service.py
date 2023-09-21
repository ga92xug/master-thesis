import subprocess
from ax import Runner
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import sys
import os
sys.path.append(f"{os.getcwd()}")

from training.main import run_experiment_from_config

# local imports
from experiments.util import convert_dict_to_hydra_string
from networks.eq_nasnet.util import encode_parameters
from new_wandb_run import create_wandb_run


class HydraWandbRunner(Runner):
    def __init__(
            self, 
            script_path: str, 
            wandb_entity: str, 
            wandb_project: str,
            wandb_mode: str, 
            db_path: str,
            choice_2_range_params: dict, 
            #strides: list, 
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
        self.db_path = db_path
        self.choice_2_range_params = choice_2_range_params
        #self.strides = strides
        self.training_dict = training_dict
        self.verbose = verbose
        
    def run(self, trial):
        trial_params, trial_index = trial

        # encode the search space parameters
        encoded_params = encode_parameters(
            params=trial_params, 
            nas_encoded=True,
            choice_2_range_params=self.choice_2_range_params
        )
        print("encoded_params", encoded_params)

        if self.verbose >= 3: 
            print("trial_params", trial_params)

        # Construct the command
        command = [self.command_prefix, self.script_path]
        # pass the encoded search space parameter
        dropout_rate = trial_params['-1_dropout_rate']
        expand_ratio = trial_params['-1_expand_ratio']
        command.extend([
            f"model=eq_nasnet",
            f"model.blocks_args_dict={convert_dict_to_hydra_string(encoded_params)}",
            f"model.dropout_rate={dropout_rate}",
            f"model.eq_expand_ratio={expand_ratio}"
        ])
        # Append all training settings
        for key, value in self.training_dict.items():
            command.append(f"{key}={value}")
                
        # pass wandb parameters
        command.extend([f"wandb.entity={self.wandb_entity}",
                        f"wandb.project={self.wandb_project}",
                        f"wandb.mode={self.wandb_mode}",
                        f"wandb.give_name=False"])

        # pass trial index
        command.extend([f"NAS.trial_index={trial_index}",
                        f"NAS.db_path={self.db_path}/trial_cache.db"])

        # run the training
        subprocess.run(command, stdout=subprocess.DEVNULL if self.verbose <= 0 else None)
        
        # Return the trial metadata
        return {
            "trial_index": trial_index,
        }

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

    
    def create_new_wandb_run(self, trial_index):
        """
        hacky way to have 2 wandb runs in the same process
        """

        wandb_params = {
            "entity": self.wandb_entity,
            "project": self.wandb_project,
            "mode": self.wandb_mode,
            "trial_index": trial_index,
        }
        # Create the command to run the script
        command = ["python", "NAS/new_wandb_run.py"]
        for key, value in wandb_params.items():
            command.extend([f"--{key}", str(value)])

        completed_process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if the script executed successfully
        if completed_process.returncode != 0:
            print(f"The script ended with an error: {completed_process.stderr.decode()}")
            raise RuntimeError("The script ended with an error")
        else:
            output = completed_process.stdout.decode()
            #print(f"The script output: {output}")

        run_id = output.split("Wandb run id: ")[-1].strip()

        return run_id
        


