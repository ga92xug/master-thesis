from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from lightning.pytorch.callbacks import BasePredictionWriter

from src.logger import pylogger
from src._callbacks.move_2_device import Move_2_Device 
from src._callbacks.log_code import Log_Code
from src._callbacks.log_config_manually import Log_Config

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks_cfg = cfg.train.get("callbacks", None)
    callbacks: List[Callback] = []

    if not callbacks_cfg or callbacks_cfg == "none":
        log.warning("No callback configs found!")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            #log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # essential callbacks
    callbacks.extend([Log_Code(), Log_Config(cfg)])

    test_mode = cfg.get("test_mode", "no_test")
    if test_mode == "predict":
        # the should be a write_predictions callback
        assert any([isinstance(callback, BasePredictionWriter) for callback in callbacks]), "No callback found for writing predictions!"

    log.info("Callbacks:", [cb.__class__.__name__ for cb in callbacks])
    return callbacks



