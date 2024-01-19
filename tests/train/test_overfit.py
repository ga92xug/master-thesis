import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import rootutils

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
from src.utils.utils import get_metric_value
from tests.train.helpers.hydra_init import hydra_compose
from src.main import main
from tests.train.helpers.run_if import RunIf

DATASETS = ["cifar10", "isic2019", "mnist", "DeepDRiD", "blood", "imagenette"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_datamodule(
    dataset: str,
    model: str = "eq_nasnet",
    num_overfit: int = 10,
    epochs: int = 100,
) -> None:
    """
    """
    
    overrides = [
        "training_setup=a_debug",
        "debug=simple",
        "test=False",
        "training_setup/callbacks=" + "metric_print",
        "training_setup/dataset=" + dataset,
        "training_setup/network=" + model,
        "training_setup.dataset.reduction_factor=" + str(num_overfit),
        "training_setup.trainer.max_epochs=" + str(epochs),
    ]

    cfg = hydra_compose(overrides)
    print(cfg)
    metric_dict, _ = main(cfg) 

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name="train/acc"
    )
    assert metric_value > 0.95, f"Training accuracy {metric_value} is too low."

if __name__ == "__main__":
    test_datamodule("imagenette", "efficientnet")
