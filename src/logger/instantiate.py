from typing import List, Union, Dict

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch
import wandb

from src.logger import pylogger
from src.networks.eq_nasnet.naming_eq_nasnet import get_scaling_name

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
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            if "wandb" in lg_conf._target_:
                # set name
                name = give_wandb_name(
                    config=cfg,
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
        if project == "SL-Scaling":
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




'''
def init_wandb(cfg: DictConfig):
    if cfg.NAS.trial_index == -1:
        # normal training mode
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        kwargs_wandb = {
            "config": wandb_config,
            "project": cfg.wandb.project,
            "mode": cfg.wandb.mode,
            "notes": cfg.wandb.notes,
            "tags": cfg.wandb.tags,
        }
        
        if cfg.ray:
            wandb_run = setup_wandb(rank_zero_only=False, **kwargs_wandb)
        else:
            wandb_run = wandb.init(project=cfg.wandb.project, config=wandb_config, \
            mode=cfg.wandb.mode, notes=cfg.wandb.notes, tags=cfg.wandb.tags)

            # merge wandb config with cfg. Sweep bug https://github.com/wandb/wandb/issues/4686
            #cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(wandb.config)))
            #wandb_update_config(cfg, wandb_run)

        wandb_run.log_code(".")
    else:
        wandb_run = None

    return wandb_run

def wandb_update_config(cfg: DictConfig, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Recursively updates the wandb config with the nested DictConfig object.
    We can not use wandb.config.update(cfg) because during sweeps some keys are not allowed to be updated.

    Args:
    - cfg (DictConfig): The nested DictConfig object.
    - wandb_run (wandb.sdk.wandb_run.Run): The wandb run object.
    """
    def recursive_update(cfg_dict, wandb_config_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                if key not in wandb_config_dict:
                    wandb_config_dict[key] = {}
                recursive_update(value, wandb_config_dict[key])
            else:
                wandb_config_dict[key] = value

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    recursive_update(cfg_dict, wandb_run.config)


        '''