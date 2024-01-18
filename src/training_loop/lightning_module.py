import signal
from typing import Any, Dict, Tuple
import hydra

import timeit
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification.accuracy import (
    Accuracy, 
    MulticlassAccuracy, 
    BinaryAccuracy
)

from src._callbacks.model_stats import get_stats, timeout_handler
from src.utils.scheduler import get_optim_and_scheduler


class LitModule(LightningModule):
    """

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: Dict[str, Any],
        optimizer: Dict[str, Any],
        scheduler: Dict[str, Any],
        num_channels: int,
        num_classes: int,
        image_size: int,
        normalization_weights: torch.Tensor,
        compile: bool,
        **kwargs: Any,
    ) -> None:
        """Initialize a `LitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = self.get_model()

        # Optimizer and scheduler
        self.optimizer, self.scheduler, self.scheduler_metric = get_optim_and_scheduler(
            self.net, optimizer, scheduler, kwargs["trainer"]["max_epochs"]
        )

        self.train_metrics = self.create_metrics_collection()
        self.valid_metrics = self.create_metrics_collection()
        self.test_metrics = self.create_metrics_collection()
        
        # Loss function
        self.criterion = CrossEntropyLoss(
            weight=normalization_weights,
            label_smoothing=kwargs.get("label_smoothing", 0),
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.valid_acc_best = MaxMetric()
        self.valid_acc_weighted_best = MaxMetric()
        self.valid_loss_best = MinMetric()

    def create_metrics_collection(self):
        metrics = {
            "acc": MulticlassAccuracy(self.hparams.num_classes, average="micro"),
            "acc_weighted": MulticlassAccuracy(self.hparams.num_classes, average="macro")
        }
        metrics_collection = MetricCollection(metrics)
        return metrics_collection


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.valid_loss.reset()
        #self.val_acc.reset()
        self.valid_metrics.reset()
        self.valid_acc_best.reset()
        self.valid_acc_weighted_best.reset()
        self.valid_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        self.log_metrics("train", loss, preds, targets)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.log_metrics("valid", loss, preds, targets)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metrics = self.valid_metrics.compute()  # get current val acc
        for key, metric in metrics.items():
            if key == "acc":
                self.valid_acc_best(metric)  # update best so far val acc
            elif key == "acc_weighted":
                self.valid_acc_weighted_best(metric)
            else:
                raise ValueError(f"Unknown metric {key}")

        loss = self.valid_loss.compute()   
        self.valid_loss_best(loss) 
        
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("valid/acc_best", self.valid_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("valid/acc_weighted_best", self.valid_acc_weighted_best.compute(), sync_dist=True, prog_bar=True)
        self.log("valid/loss_best", self.valid_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.log_metrics("test", loss, preds, targets)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": self.scheduler_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": self.optimizer}
    

    def log_metrics(self, mode: str, loss, preds, targets) -> None:
        """
        Log metrics for a given mode (train, valid, test).

        Args:
        - mode (str): The mode for logging ('train', 'valid', 'test').
        - loss: The computed loss for the current batch.
        - preds: The predictions made by the model.
        - targets: The actual targets/labels.
        """
        if mode not in ['train', 'valid', 'test']:
            raise ValueError("Mode must be 'train', 'valid', or 'test'.")

        loss_func = getattr(self, f"{mode}_loss")
        metrics_func = getattr(self, f"{mode}_metrics")

        # Update and log loss
        loss_func(loss)
        self.log(f"{mode}/loss", loss_func, on_step=False, on_epoch=True, prog_bar=True)

        # Update and log metrics
        metrics_func(preds, targets)
        for key, metric in metrics_func.items():
            self.log(f"{mode}/{key}", metric, on_step=False, on_epoch=True, prog_bar=True)

    

    def get_model(
        self,
        verbose: int = 1,
        is_nas: bool = False,
    ):
        """
        Instantiate the model and return:
        - number of parameters
        - model building time
        - train time
        - GFLOPs
        """
        num_channels = self.hparams.num_channels
        num_classes = self.hparams.num_classes
        image_size = self.hparams.image_size

        if is_nas:
            # Set the maximum allowed execution time in seconds
            max_building_time = self.hparams.NAS.max_building_time
            max_gflops = self.hparams.NAS.max_gflops

            assert max_building_time > 0, "max_building_time must be greater than 0"
            self.log({"model_building_time": max_building_time}, verbose=2)
            assert max_gflops > 0, "max_gflops must be greater than 0"

            # Set the signal handler for the timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max_building_time)


        # create model
        net = None
        start = timeit.default_timer()
        net = hydra.utils.instantiate(
                self.hparams.network,
                num_channels=num_channels,
                num_classes=num_classes,
                image_size=image_size,
                verbose=verbose,
            )
        stop = timeit.default_timer()
        self.net_building_time = stop - start

        if is_nas:
            # Cancel alarm
            signal.alarm(0)
        
        return net



if __name__ == "__main__":
    _ = LitModule(None, None, None, None)
