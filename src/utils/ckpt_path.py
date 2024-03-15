from lightning import Trainer
from omegaconf import DictConfig
import os

from src.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


def get_ckpt_path(cfg: DictConfig, mode: str, trainer: Trainer) -> str:
    """Returns the path to the best checkpoint for testing.

    :param cfg: A DictConfig configuration composed by Hydra.
    :param trainer: A Lightning Trainer object.
    :return: The path to the best checkpoint.

    :raises RuntimeError: If no checkpoint is found or no ckpt provided for eval only mode.
    """
    
    if mode == "train":
        # normal training
        # no ckpt path needed
        raise RuntimeError("No ckpt path needed for train mode!")
    elif mode == "evaluate_only":
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is None:
            raise RuntimeError("No ckpt path provided for evaluate_only mode!")
    elif mode == "train_continue":
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is None:
            raise RuntimeError("No ckpt path provided for train_continue mode!")
    elif mode in ["test", "predict"]:
        if trainer.checkpoint_callback is None:
            log.warning(f"No checkpoint found for {mode}. Using current weights.")
            ckpt_path = ""
        else:
            ckpt_path = trainer.checkpoint_callback.best_model_path

    # 
    if ckpt_path == "":
        return None
    else:
        if not os.path.exists(ckpt_path):
            # if the ckpt_path does not exist we try to find it in the log_dir
            ckpt_path = os.path.join(cfg.paths.log_dir, ckpt_path)
            print("ckpt", ckpt_path)
            if not os.path.exists(ckpt_path):
                raise RuntimeError(f"Checkpoint {ckpt_path} not found!")

    return ckpt_path


def save_ckpt_during_training(trainer: Trainer):
    if trainer.checkpoint_callback is None:
        return ""
    else:
        return trainer.checkpoint_callback.best_model_path