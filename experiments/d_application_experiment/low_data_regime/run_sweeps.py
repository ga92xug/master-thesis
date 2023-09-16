from typing import Dict
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

# 500 epochs 1% data - 100 epochs 100% data

def prepare_sweep_config(
        config_dict: Dict,
        percentage: int,
        num_of_epochs: int,

    ):
    # extract sweep config
    sweep_config = config_dict["sweep_config"]

    # create command 
    command = sweep_config["command"]
    name = sweep_config["name"]
    epochs_at_100 = sweep_config["epochs"]
    epochs_at_current = get_number_epochs(percentage, epochs_at_100)
    print("command", command)
    extend_command = [
        f"training={name}-training",
        f"training.epochs={epochs_at_current}",
        f"training.dataset.reduction_factor={percentage / 100}",
        f"training.scheduler=cosine_annealing_lr",
    ]
    sweep_config["command"] = command + " ".join(extend_command)

    # early stopping at 10% of epochs
    sweep_config["early_terminate"]["min_iter"] = get_min_iter(epochs_at_current)

    # sweep name
    sweep_config["name"] = sweep_config["name"] + "_low_data_" + percentage

    # description
    sweep_config["description"] = sweep_config["name"] + "_low_data_" + percentage + "_with_epochs_" + epochs_at_current

    return sweep_config

def get_number_epochs(percentage: float, epochs_without_reduction: int):
    return int((1/ (percentage / 100)) * epochs_without_reduction)

def get_min_iter(epochs: int):
    """
    Grace period for early stopping
    """
    return int(0.1 * epochs)

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    percentage = 1
    sweep_config = prepare_sweep_config(config_dict, percentage)

    # init sweep agent
    sweep_id = wandb.sweep(sweep_config)
    # run sweep agent
    wandb.agent(sweep_id, count=cfg.num_trials)