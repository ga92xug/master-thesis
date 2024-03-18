import subprocess
from typing import Any, Dict, List, Tuple
from ax import Runner
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import sys
import os

sys.path.append(f"{os.getcwd()}")
from src.optimization.optimization_utils import convert_dict_to_hydra_string


class Hydra_Submitter(Runner):
    def __init__(
        self,
        additional_overrides: Dict[str, Any] = None,
        script_path: str = "src/main.py", 
        verbose: int = 10,
    ):
        super().__init__()
        self.script_path = script_path
        self.additional_overrides = additional_overrides
        self.verbose = verbose
        
    def construct_command(self, trial: Tuple[Dict[str, Any], int]) -> List[str]:
        # Unpack the trial
        trial_params, trial_index = trial
        # Construct the command
        command = ["python", self.script_path]
        # Append all additional settings
        for key, value in self.additional_overrides.items():
            command.append(f"{key}={value}")

        command.extend([
            f"NAS.trial_index={trial_index}",
            f"logger.csv.version={trial_index}"
        ])

        # pass wandb parameters
        #command.extend([f"wandb.project={self.wandb_project}",
        #                f"wandb.mode={self.wandb_mode}",
        #                f"wandb.give_name=False"])

        print("trial_params", trial_params)
        # encode the search space parameters
        encoded_params = encode_parameters(
            params=trial_params, 
            nas_encoded=True,
        )
        print("encoded_params", encoded_params)

        
        # pass the encoded search space parameter
        dropout_rate = trial_params['-1_dropout_rate']
        expand_ratio = trial_params['-1_expand_ratio']
        command.extend([
            f"train.network.blocks_args_dict={convert_dict_to_hydra_string(encoded_params)}",
            f"train.network.dropout_rate={dropout_rate}",
            f"train.network.eq_expand_ratio={expand_ratio}"
        ])
        return command

    def run(self, trial):
        command = self.construct_command(trial)
        
        # run the training
        subprocess.run(command, stdout=subprocess.DEVNULL if self.verbose <= 0 else None)
        
        # Return the trial metadata
        return {"trial_index": trial[1]}
    
def encode_parameters(
        params: Dict[str, Any], 
        nas_encoded: bool = True, 
        choice_2_range_params : Dict[str, List[int]] = {"group": [1, 2, 4, 8, 16]},
        nas_key_2_eq_nasnet_key: Dict[str, str] = {
            # the names changed slightly
            "skip_op": "skip",
            "out_channels": "out_channel",  
            # the names that did not change
            "expand_ratio": "-1_expand_ratio",
            "dropout_rate": "-1_dropout_rate",
            "reflection": "reflection",
            "group": "group",
            "num_layers": "num_layers",
            "conv_op": "conv_op",
            "kernel_size": "kernel_size",
            "se_ratio": "se_ratio",
            "stride": "stride",

        },
    ):
    """
    Encodes the parameters into a dict representation. If group_encoded is True, the group parameter is encoded as a number from 0 to 4, otherwise it is encoded as a number from 1 to 16.

    Args:
        params (dict): A dictionary containing the parameters.
        choice_2_range_params (dict): A dictionary containing the discrete 
            choices for some of the params.

    Returns:
        dict: The encoded dict representation of the parameters.
    """
    # Number of blocks is determined by the highest numbered block in the keys of params

    num_blocks = max(int(key.split('_')[0]) for key in params.keys() if key.split('_')[0].isdigit()) + 1
    
    blocks = {f"_{i}": {} for i in range(num_blocks)}

    for key, value in params.items():
        block_index, param_name = key.split("_")[0], "_".join(key.split("_")[1:])

        if param_name in nas_key_2_eq_nasnet_key:
            # convert the param name to the nasnet equivalent 
            # the names changed slightly
            param_name = nas_key_2_eq_nasnet_key[param_name]
        else:
            print(f"key {key} not found in nas_key_2_eq_nasnet_key")
            continue  

        if block_index == "-1":
            # expand_ratio and dropout_rate are not in a block
            pass
            #blocks[key] = value

        elif block_index.isdigit():
            block_index = "_" + str(int(block_index))
            if param_name == "group" and nas_encoded:
                # group is encoded as a number from 0 to 4 
                value = choice_2_range_params['group'][int(value)]
                
            blocks[block_index][param_name] = value
        else:
            raise ValueError(f"block_index should be a number, got {block_index}, type: {type(block_index)}, key: {key}, value: {value}")

    return blocks