import signal
from typing import Any, Dict, Tuple
import hydra

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification.accuracy import (
    Accuracy, 
    MulticlassAccuracy, 
    BinaryAccuracy
)

from src.training_loop.utils import get_stats, init_model, timeout_handler


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

        self.net = self.get_model(
            num_channels=num_channels,
            num_classes=num_classes,
            image_size=image_size,
        )

        # Optimizer and scheduler
        self.optimizer = hydra.utils.instantiate(optimizer, params=self.net.parameters())
        self.scheduler_metric = scheduler.get("metric", None)
        del scheduler.metric
        self.scheduler = hydra.utils.instantiate(scheduler, optimizer=self.optimizer)

        self.train_metrics = self.create_metrics_collection()
        self.valid_metrics = self.create_metrics_collection()
        self.test_metrics = self.create_metrics_collection()
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(weight=normalization_weights)

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

        # update and log metrics
        self.train_loss(loss)
        self.train_metrics(preds, targets)
        #self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        for key, metric in self.train_metrics.items():
            self.log(f"train/{key}", metric, on_step=False, on_epoch=True, prog_bar=True)

        #self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

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

        # update and log metrics
        self.valid_loss(loss)
        #self.val_acc(preds, targets)
        self.valid_metrics(preds, targets)
        self.log("valid/loss", self.valid_loss, on_step=False, on_epoch=True, prog_bar=True)
        for key, metric in self.valid_metrics.items():
            self.log(f"valid/{key}", metric, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

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

        # update and log metrics
        self.test_loss(loss)
        #self.test_acc(preds, targets)
        self.test_metrics(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        for key, metric in self.test_metrics.items():
            self.log(f"test/{key}", metric, on_step=False, on_epoch=True, prog_bar=True)

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
    

    def get_model(
        self,
        net_cfg: DictConfig, 
        num_channels: int, 
        num_classes: int, 
        image_size: int,
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
        net, net_building_time = init_model(
            net_cfg, 
            num_channels, 
            num_classes, 
            image_size, 
            verbose
        )
        if is_nas:
            # Cancel alarm
            signal.alarm(0)
        
        self.log({"net_building_time": net_building_time})


        gflops_per_image, param_count = get_stats(
                net, 
                self.hparams.data, 
                num_channels, 
                image_size
            )
        self.log({"GFLOPs_per_image": gflops_per_image})
        self.log({"param_count": param_count})

        if is_nas and gflops_per_image > max_gflops:
            raise ValueError(f"GFLOPs {gflops_per_image} exceeds maximum allowed {max_gflops}")

        return net



if __name__ == "__main__":
    _ = LitModule(None, None, None, None)
