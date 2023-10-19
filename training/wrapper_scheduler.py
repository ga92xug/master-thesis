from typing import Dict
import hydra
from omegaconf import open_dict

from training.logger import Custom_Logger


class Wrapper_Scheduler:
    def __init__(self, cfg, optimizer, dataloaders: Dict, logger: Custom_Logger):
        self.scheduler = None
        self.metric_lr_in_validation = None
        self.optimizer = optimizer
        self.logger = logger

        if cfg.training.scheduler._target_ is not None:
            # adapt learning rate
            if "CosineAnnealingLR" in cfg.training.scheduler._target_:
                with open_dict(cfg):
                    cfg.training.scheduler.T_max = len(dataloaders["train"]) * cfg.training.epochs

            if "ReduceLROnPlateau" in cfg.training.scheduler._target_: 
                self.metric_lr_in_validation = cfg.training.scheduler.metric
                assert self.metric_lr_in_validation.split(".")[0] == "valid", f"metric {self.metric_lr_in_validation} is not a valid metric for ReduceLROnPlateau"
                self.metric_lr_in_validation = self.metric_lr_in_validation.split(".")[1]
                del cfg.training.scheduler.metric # remove the metric from the scheduler config since it is not a valid argument

            self.scheduler = hydra.utils.instantiate(cfg.training.scheduler, 
                                                optimizer=optimizer)

    def step_epoch_end(self):
        """
        update learning rate if learning rate is not adapted in validation
        """
        # update learning rate if learning rate is not adapted in validation
        if self.metric_lr_in_validation is None and self.scheduler is not None:
            self.scheduler.step()

    def step_validation(self, metrics):
        """
        update learning rate if learning rate is adapted in validation
        """
        # update learning rate if learning rate is adapted in validation
        if self.metric_lr_in_validation is not None and self.scheduler is not None:
            metric = metrics.get(self.metric_lr_in_validation, None)
            if metric is None:
                raise ValueError(f"metric {self.metric_lr_in_validation} not found in metrics {metrics}")

            # adapt lr    
            self.scheduler.step(metric)

            # log the current learning rate
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.log({"scheduler": {"lr": lr, "epoch": self.logger.epoch.value}}, verbose=2)

            