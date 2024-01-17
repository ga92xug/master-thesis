import signal
from typing import List
import wandb
import hydra
from hydra import compose, initialize

def hydra_compose(overrides: List[str]):
    with initialize(config_path="../../../configs", version_base="1.3"):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    return cfg