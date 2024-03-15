import os
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
import wandb
from omegaconf import DictConfig
from wandb.sdk.lib import filenames

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class Prediction_txt_Writer(BasePredictionWriter):
    def __init__(self, output_dir, write_interval='batch'):
        super().__init__(write_interval)
        print("output_dir: ", output_dir)
        self.output_dir = os.path.join(output_dir, 'submission.txt')

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        The predictions should be the logits.
        The batch_indices should be the image_ids which are monotonically increasing.
        """
        if trainer.global_rank != 0:
            raise NotImplementedError("Prediction_txt_Writer is only supported with `Trainer(gpus=0)`.")
        
        # Process and save predictions
        with open(self.output_dir, 'w') as file:
            file.write(str(prediction.tolist()) + '\n')

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):  
        # Process and save predictions
        with open(self.output_dir, 'w') as file:
            for prediction in predictions:
                # Format the prediction as needed (e.g., image_id, predicted_label)
                file.write(str(prediction.tolist()) + '\n')

