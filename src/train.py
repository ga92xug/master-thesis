from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import os
os.environ['HYDRA_FULL_ERROR'] = '1'

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
from src.data.datamodule import DataModule

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    log_hyperparameters,
    task_wrapper,
)

from src.logger import (
    instantiate_loggers
)

from src.adversarial_attack.adversarial_attack import adversarial_attack

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate(
    cfg: DictConfig
) -> Dict[str, Union[DictConfig, DataModule, LightningModule, List[Callback], List[Logger], Trainer]]:
    """
    Instantiates all objects needed for lightning training (or testing).
    """
    
    is_dist = True if cfg.hardware.devices > 1 or cfg.hardware.num_nodes > 1 else False

    log.info("Instantiating datamodule")
    datamodule = DataModule(cfg.training_setup.dataset, is_dist=is_dist)

    log.info("Instantiating model")
    model = LightningModule(
        **cfg.training_setup,
        num_channels=datamodule.num_channels,
        num_classes=datamodule.num_classes,
        image_size=datamodule.image_size,
        normalization_weights=datamodule.normalization_weights,
    )

    log.info("Instantiating callbacks")
    callbacks: List[Callback] = instantiate_callbacks(cfg.training_setup.get("callbacks"))
    log.info(callbacks)

    log.info("Instantiating loggers")
    logger: List[Logger] = instantiate_loggers(
        cfg=cfg,
        model_name=model.net.name
    )

    log.info(f"Instantiating trainer")
    trainer = Trainer(
        **{**cfg.training_setup.trainer, **cfg.hardware},
        callbacks=callbacks,
        logger=logger,
        # distributed sampling is already done by our datamodule
        use_distributed_sampler=False,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    return object_dict


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    object_dict = instantiate(cfg)
    datamodule: DataModule = object_dict["datamodule"]
    model: LightningModule = object_dict["model"]
    callbacks: List[Callback] = object_dict["callbacks"]
    logger: List[Logger] = object_dict["logger"]
    trainer: Trainer = object_dict["trainer"]  

    if cfg.get("train"):
        assert not cfg.get("eval_only", False), "No training in eval only mode"
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        if cfg.get("eval_only"):
            ckpt_path = cfg.get("ckpt_path")
        else:
            ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        if isinstance(trainer.strategy, DDPStrategy):
            # set number of devices and nodes to 1 for testing
            trainer = Trainer(
                **{**cfg.training_setup.trainer, **cfg.hardware},
                callbacks=callbacks,
                logger=logger,
                # distributed sampling is already done by our datamodule
                use_distributed_sampler=False,
                num_nodes=1,
                devices=1,
                strategy="auto"
            )

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    if cfg.get("adversarial_attack", False):
        log.info("Start adversarial attack")
        datamodule.setup()
        adversarial_attack(
            mode=cfg.adversarial_attack,
            model=model.net,
            dataloader=datamodule.test_dataloader(),
            cfg=cfg,
            logger=logger,
        )


    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg) 

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
