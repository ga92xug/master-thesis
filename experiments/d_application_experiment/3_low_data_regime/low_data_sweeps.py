from typing import Dict
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import os
os.environ['HYDRA_FULL_ERROR'] = '1'

# 500 epochs 1% data - 100 epochs 100% data

def prepare_sweep_config(
        config_dict: dict,
    ):
    percentage = config_dict["percentage"]
    sweep_config = config_dict["sweep_config"]

    # extend command 
    sweep_config["command"].extend([
        f"training.dataset.reduction_factor={percentage / 100}",
    ])

    # early stopping at 10% of epochs
    epochs = sweep_config.pop("epochs") # remove epochs from sweep config no other keys allowed
    sweep_config["early_terminate"]["min_iter"] = int(0.1 * epochs)

    # sweep name + description
    sweep_config["name"] = sweep_config["name"] + "_low_data_" + str(percentage)
    sweep_config["description"] = sweep_config["name"] + "_with_epochs_" + str(epochs)
    return sweep_config


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    sweep_config = prepare_sweep_config(config_dict)

    # init sweep agent
    sweep_id = wandb.sweep(sweep_config)
    # run sweep agent
    wandb.agent(sweep_id)


if __name__ == "__main__":
    main()