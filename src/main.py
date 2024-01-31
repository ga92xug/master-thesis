from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from lightning.pytorch.callbacks import BasePredictionWriter

import os
os.environ['HYDRA_FULL_ERROR'] = '1'

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
from src.data.datamodule import DataModule
from src.training_loop.lightning_module import LitModule
from src.utils.ckpt_path import get_ckpt_path

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    log_hyperparameters,
    task_wrapper,
    get_test_trainer,
)

from src.logger import (
    instantiate_loggers
)

from src.adversarial_attack.adversarial_attack import adversarial_attack

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate(
    cfg: DictConfig,
    train_mode: str,
) -> Dict[str, Union[DictConfig, DataModule, LightningModule, List[Callback], List[Logger], Trainer]]:
    """
    Instantiates all objects needed for lightning training (or testing).
    """
    
    is_dist = True if cfg.hardware.get("devices", 1) > 1 or cfg.hardware.get("num_nodes", 1) > 1 else False

    log.info("Instantiating datamodule")
    datamodule = DataModule(cfg.training_setup.dataset, is_dist=is_dist)

    log.info("Instantiating model")
    if train_mode == "train_with_ckpt":
        # load pretrained model
        log.info("Loading pretrained model")
        ckpt_path = get_ckpt_path(cfg, train_mode, trainer=None)
        assert ckpt_path, "Checkpoint path must be provided for train_with_ckpt mode."
        model = LitModule.load_from_checkpoint(ckpt_path)
    else:
        # for the other modes the trainer will take care of loading the weights if needed
        model = LitModule(
            **cfg.training_setup,
            num_channels=datamodule.num_channels,
            num_classes=datamodule.num_classes,
            image_size=datamodule.image_size,
            normalization_weights=datamodule.normalization_weights,
        )

    log.info("Instantiating callbacks")
    callbacks: List[Callback] = instantiate_callbacks(cfg.training_setup.get("callbacks"))
    test_mode = cfg.get("test_mode", "no_test")
    if test_mode == "predict":
        # the should be a write_predictions callback
        assert any([isinstance(callback, BasePredictionWriter) for callback in callbacks]), "No callback found for writing predictions!"
    

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

    train_mode = cfg.get("train_mode", "train")

    object_dict = instantiate(cfg, train_mode)
    datamodule: DataModule = object_dict["datamodule"]
    model: LightningModule = object_dict["model"]
    callbacks: List[Callback] = object_dict["callbacks"]
    logger: List[Logger] = object_dict["logger"]
    trainer: Trainer = object_dict["trainer"]  
    
    if train_mode == "evaluate_only":
        log.info("Running in evaluate_only mode.")
        ckpt_path = get_ckpt_path(cfg, train_mode, trainer)

        new_model = model.__class__(**model.hparams)
        state_dict_new = torch.load("temp_model.pth")
        print_keys(state_dict_new)
        
        new_model.load_state_dict(state_dict_new)
        out = trainer.validate(model=new_model, datamodule=datamodule)
        print("New Val out", out)

        
        
        out = trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        print("Normal Val out", out)

        
    elif train_mode in ["train", "train_with_ckpt"]:
        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)
    elif train_mode == "train_continue":
        log.info("Continuing training from checkpoint.")
        ckpt_path = get_ckpt_path(cfg, train_mode, trainer)
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Unknown training mode: {train_mode}")

    train_metrics = trainer.callback_metrics
    test_mode = cfg.get("test_mode", "no_test")

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="conf.yaml")
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
