from typing import List, Union, Dict

import hydra
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.logger import pylogger
log = pylogger.RankedLogger(__name__, rank_zero_only=True)



def instantiate_loggers(
    cfg: DictConfig, 
    model_name: Union[str, Dict[str, str]],
) -> List[Logger]:
    """Instantiates loggers from config.

    :param cfg: A DictConfig object containing logger configurations.
    :param model_name: The name of the model.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    logger_cfg = cfg.get("logger")

    if not logger_cfg:
        log.warning("No logger configs found!")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            if "wandb" in lg_conf._target_:
                # set name
                name = give_wandb_name(
                    config=lg_conf,
                    model_name=model_name,
                )
                lg_conf["name"] = name

            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def give_wandb_name(
        config: DictConfig,
        model_name: Union[str, Dict[str, str]],
    ) -> str:
    """
    Updates the wandb_run name.
    """

    give_name = config.get("name", None)
    project = config["project"]

    if give_name is None:
        return None
    
    # do check for true, since strings are also true
    if give_name == True:
        if project == "scaling":
            assert isinstance(model_name, dict), "for scaling name need a dict"
            return model_name["scaling_name"]
        else:
            if isinstance(model_name, str):
                return model_name
            else:
                return model_name["model_name"]
    elif isinstance(give_name, str):
        return give_name
    else:
        raise ValueError("give_name must be True or a string!")


