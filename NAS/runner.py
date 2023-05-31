import os
import subprocess
from ax import Runner
import wandb
from omegaconf import OmegaConf

from util import encode_parameters


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
        # Construct the command
        command = [self.command_prefix, self.script_path]

        # Add the trial parameters
        trial_index = trial.index
        print(trial)
        arm = trial.arm
        trial_params = arm.parameters

        # Create a new wandb run
        run = wandb.init(
            entity=self.wandb_entity,
            project=self.wandb_project, 
            mode=self.wandb_mode,
            name=str(trial_index),
        )

        # encode the search space parameters
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
                        f"wandb.run_id={run.id}",
                        f"wandb.mode={self.wandb_mode}"])

        # run the training
        subprocess.run(command, stdout=subprocess.DEVNULL if self.verbose <= 0 else None)

        # Return the trial metadata
        trial_metadata = {
            "name": str(trial_index),
            "wandb_run_id": run.id,
        }

        return trial_metadata


    def poll_trial_status(
        self, trials
    ):
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Scheduler``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        status_dict = defaultdict(set)
        for trial in trials:
            mock_job_queue = get_mock_job_queue_client()
            status = mock_job_queue.get_job_status(
                job_id=trial.run_metadata.get("job_id")
            )
            status_dict[status].add(trial.index)

        return status_dict


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

    
