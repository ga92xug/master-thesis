from typing import Any, Dict, List

from lightning_utilities.core.rank_zero import rank_zero_only
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.logger import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(loggers: List[Logger], cfg: DictConfig, net_building_time: str) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(cfg, resolve=True)

    if not loggers and len(loggers) == 0:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["network"] = cfg["train"]["network"]
    hparams["dataset"] = cfg["train"]["dataset"]
    hparams["optimizer"] = cfg["train"]["optimizer"]
    hparams["scheduler"] = cfg["train"].get("scheduler")
    hparams["callbacks"] = cfg["train"].get("callbacks")
    hparams["trainer"] = cfg["train"]["trainer"]
    if cfg.get("training"):
        # backward compatibility
        hparams["training"] = cfg["training"]

    hparams["hardware"] = cfg["hardware"]
    hparams["extras"] = cfg.get("extras")
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    hparams["optimized_metric"] = cfg.get("optimized_metric")
    hparams["net_building_time"] = net_building_time

    # send hparams to all loggers
    for logger in loggers:
        logger.log_hyperparams(hparams)