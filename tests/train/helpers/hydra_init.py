import signal
from typing import List
import wandb
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig


def hydra_compose(overrides: List[str]):
    initialize(config_path="../../../configs", version_base="1.3") # :
    cfg = compose(config_name="train.yaml", overrides=overrides, return_hydra_config=True)
    HydraConfig().set_config(cfg)
    return cfg