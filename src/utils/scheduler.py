from typing import Any, Dict, Tuple
import hydra
import torch
from omegaconf import DictConfig, open_dict


def get_optim_and_scheduler(
    net: torch.nn.Module,
    optimizer: DictConfig,
    scheduler: DictConfig,
    max_epochs: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, str]:
    """ Instantiates optimizer and scheduler from config.
        Adapted from the torch vision train script: 
        https://github.com/pytorch/vision/blob/main/references/classification/train.py#L304
    """

    optimizer = hydra.utils.instantiate(optimizer, params=net.parameters())
    if scheduler is not None and len(scheduler) > 0 and scheduler._target_ is not None:
        with open_dict(scheduler):
            scheduler_metric = scheduler.pop("metric", None)
            warmup_epochs = scheduler.pop("warmup_epochs", 0)
            warmup_decay = scheduler.pop("warmup_decay", 0.1)

        if "CosineAnnealingLR" in scheduler._target_:
            # T_max has to be calculated 
            with open_dict(scheduler):
                scheduler.T_max = max_epochs - warmup_epochs
            
        main_lr_scheduler = hydra.utils.instantiate(scheduler, optimizer=optimizer)

        if warmup_epochs > 0:
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_decay, total_iters=warmup_epochs
            )

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epochs]
            )
        else:
            scheduler = main_lr_scheduler

    else:
        scheduler = None
        scheduler_metric = None


    return optimizer, scheduler, scheduler_metric