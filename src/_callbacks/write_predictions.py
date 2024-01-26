import os
from typing import Callable, List, Optional, Tuple, Union
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
import wandb
from omegaconf import DictConfig
from wandb.sdk.lib import filenames

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class Prediction_txt_Writer(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = os.path.join(output_dir, 'submission.txt')

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):  
        print("Writing predictions", self.output_dir)
        print("trainer rank", trainer.global_rank)
        print("predictions: ", predictions)
        # Process and save predictions
        with open(self.output_dir, 'w') as file:
            for prediction in predictions:
                # Format the prediction as needed (e.g., image_id, predicted_label)
                file.write(prediction + '\n')
