from typing import Any, Dict, List, Optional, Tuple, Union
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import os
import random
torch.set_float32_matmul_precision('high')
os.environ['HYDRA_FULL_ERROR'] = '1'

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
from src.data.datamodule import DataModule
from src.training_loop.lightning_module import LitModule
from src.utils.ckpt_path import get_ckpt_path
from src.logger import instantiate_loggers
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    log_hyperparameters,
    task_wrapper,
    get_test_trainer,
)
log = RankedLogger(__name__, rank_zero_only=True)


def instantiate(
    cfg: DictConfig,
    train_mode: str,
) -> Tuple[DataModule, LightningModule, List[Callback], List[Logger], Trainer]:
    """
    Instantiates all objects needed for lightning training (or testing).
    """
    
    is_dist = True if cfg.hardware.get("devices", 1) > 1 or cfg.hardware.get("num_nodes", 1) > 1 else False

    log.info("Instantiating datamodule")
    datamodule = DataModule(cfg.train.dataset, is_dist=is_dist)

    log.info("Instantiating model")
    model = LitModule(
        **cfg.train,
        num_channels=datamodule.num_channels,
        num_classes=datamodule.num_classes,
        image_size=datamodule.image_size,
        normalization_weights=datamodule.normalization_weights,
        seed=cfg.seed,
    )

    log.info("Instantiating callbacks")
    callbacks: List[Callback] = instantiate_callbacks(cfg)

    log.info("Instantiating loggers")
    logger: List[Logger] = instantiate_loggers(
        cfg=cfg,
        model_name=model.net.name
    )

    log.info(f"Instantiating trainer")
    trainer = Trainer(
        **{**cfg.train.trainer, **cfg.hardware},
        callbacks=callbacks,
        logger=logger,
        use_distributed_sampler=False, # distributed sampling is already done by our datamodule
    )

    log_hyperparameters(loggers=logger, cfg=cfg)
    return datamodule, model, callbacks, logger, trainer

def train(
        cfg: DictConfig,
        datamodule: DataModule,
        model: LightningModule,
        trainer: Trainer,
        train_mode: str,
    ) -> Dict[str, Any]:
    if train_mode == "evaluate_only":
        log.info("Running in evaluate_only mode.")
        ckpt_path = get_ckpt_path(cfg, train_mode, trainer)
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif train_mode in ["train", "train_with_pretrain"]:
        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)
    elif train_mode == "train_continue":
        log.info("Continuing training from checkpoint.")
        ckpt_path = get_ckpt_path(cfg, train_mode, trainer)
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Unknown training mode: {train_mode}")
    
    return trainer.callback_metrics

def test(
        cfg: DictConfig,
        datamodule: DataModule,
        model: LightningModule,
        trainer: Trainer,
        test_mode: str,
        logger: List[Logger],
        callbacks: List[Callback],
    ):
    if test_mode == "test":
        log.info("Starting testing.")
        trainer = get_test_trainer(cfg, trainer, logger=logger, callbacks=callbacks)
        ckpt_path = get_ckpt_path(cfg, test_mode, trainer)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif test_mode == "predict":
        log.info("Starting prediction.")
        trainer = get_test_trainer(cfg, trainer, logger=logger, callbacks=callbacks)
        ckpt_path = get_ckpt_path(cfg, test_mode, trainer)
        # predictions are saved in the callback
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path, return_predictions=False)
    elif test_mode == "no_test":
        log.info("Skipping testing.")
    else:
        raise ValueError(f"Unknown test mode: {test_mode}")

    return trainer.callback_metrics

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # instantiate all objects
    train_mode = cfg.get("train_mode", "train")
    datamodule, model, callbacks, logger, trainer = instantiate(cfg, train_mode) 
    
    # train
    train_metrics = train(cfg, datamodule, model, trainer, train_mode)

    # test
    test_mode = cfg.get("test_mode", "no_test")
    test_metrics = test(cfg, datamodule, model, trainer, test_mode, logger, callbacks)

    # adversarial attack
    if cfg.get("adversarial_attack", False):
        log.info("Start adversarial attack")
        datamodule.setup()
        hydra.utils.call(
            cfg.adversarial_attack, 
            model=model.net,
            dataloader=datamodule.test_dataloader(), 
            cfg=cfg, 
            logger=logger
        )
        
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="conf.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    if cfg.get("test_hp_search", False):
        # if we are testing the hyperparameter search, we return a random value
        return random.random()
    L.seed_everything(cfg.seed, workers=True)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict = train(cfg) 

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()

