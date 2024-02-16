import copy
import signal
from typing import Any, Dict, List, Tuple, Union, Mapping
import hydra

import timeit
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Module
from lightning import LightningModule
import lightning as L
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification.accuracy import (
    Accuracy, 
    MulticlassAccuracy, 
    BinaryAccuracy
)
from torchmetrics.wrappers import MetricTracker

from src._callbacks.model_stats import timeout_handler
from src.training_loop.utils import create_metrics_collection
from src.utils.equivariant_utils import is_equivariant_model, create_filters_network
from src.utils.scheduler import get_optim_and_scheduler
from src.utils.utils import recursive_print_dict


class LitModule(LightningModule):
    def __init__(
        self,
        network: Dict[str, Any],
        optimizer: Dict[str, Any],
        num_channels: int,
        num_classes: int,
        image_size: int,
        normalization_weights: torch.Tensor,
        compile: bool,
        seed: int,
        metrics_config: DictConfig = None,
        scheduler: Dict[str, Any] = None,
        label_smoothing: float = 0,
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
        
        # Loss function
        self.criterion = CrossEntropyLoss(
            weight=normalization_weights,
            label_smoothing=label_smoothing,
        )
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.valid_loss_best = MinMetric()

        # Metrics 
        self.train_metrics = create_metrics_collection(num_classes, metrics_config)
        self.valid_metrics: MetricTracker = create_metrics_collection(num_classes, metrics_config, valid=True)
        self.test_metrics = create_metrics_collection(num_classes, metrics_config)

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
        self.valid_metrics.reset_all()
        self.valid_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of logits.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        #logits = torch.argmax(logits, dim=1)
        return loss, logits, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, targets = self.model_step(batch)
        self.log_metrics("train", loss, logits, targets)
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
        loss, logits, targets = self.model_step(batch)
        self.log_metrics("valid", loss, logits, targets)

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        #if self.trainer.sanity_checking:
        #    return
        self.valid_metrics.increment()

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if self.trainer.sanity_checking:
            # sanity check does not have a best metric
            return
        loss = self.valid_loss.compute()   
        self.valid_loss_best(loss)
        self.log("valid_best/loss", self.valid_loss_best.compute(), sync_dist=True, prog_bar=False) 

        best_metrics = self.valid_metrics.best_metric()
        for key, metric in best_metrics.items():
            self.log(f"valid_best/{key}", metric, on_step=False, on_epoch=True, prog_bar=False)
        

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits, targets = self.model_step(batch)
        self.log_metrics("test", loss, logits, targets)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    
    def predict_step(self, batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: Can be a tuple of (x, y) or just x.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        """
        if isinstance(batch, torch.Tensor):
            x = batch
        elif isinstance(batch, tuple) or isinstance(batch, list):
            # usually in predict we have no labels but useful for testing predict
            x, _ = batch
        else:
            raise ValueError(f"Unknown batch type {type(batch)}")
        
        logits = self.net(x)
        return logits

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
    

    def log_metrics(self, mode: str, loss: torch.Tensor, logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> None:
        """
        Log metrics for a given mode (train, valid, test).

        Args:
        - mode (str): The mode for logging ('train', 'valid', 'test').
        - loss: The computed loss for the current batch.
        - logits: The predictions made by the model.
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
        metrics_func(logits, targets)
        metrics_dict = metrics_func.compute()
        for key, metric in metrics_dict.items():
            self.log(f"{mode}/{key}", metric, on_step=False, on_epoch=True, prog_bar=True)

    def get_model(
        self,
        verbose: int = 1,
        is_nas: bool = False,
    ) -> torch.nn.Module:
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
        net: torch.nn.Module = hydra.utils.instantiate(
                self.hparams.network,
                num_channels=num_channels,
                num_classes=num_classes,
                image_size=image_size,
                #verbose=verbose,
            )
        stop = timeit.default_timer()
        self.net_building_time = stop - start

        if is_nas:
            # Cancel alarm
            signal.alarm(0)

        # the model has to be in eval mode for loading the weights
        # net.eval()        
        return net

   
    ############################################################################
    # Methods for loading and saving the model
    ############################################################################    

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if not is_equivariant_model(self.net):
            # if the model is not equivariant, we can load the state dict directly
            super().load_state_dict(state_dict, strict)
            return
        
        # if the model is equivariant, we have to do some extra work
        is_train_mode = self.net.training
        # load in eval mode
        if is_train_mode:
            self.net.eval()

        super().load_state_dict(state_dict, strict)        
        #create_filters_network(self.net)
        # put back in train mode
        if is_train_mode:
            self.net.train()

    def on_save_checkpoint(self, checkpoint):
        if not is_equivariant_model(self.net):
            # if the model is not equivariant nothing to do
            return 
        # save to seed that was used to initialize the model
        checkpoint['seed'] = self.hparams.seed
        
        # filters are destroyed before the model is saved
        #create_filters_network(self.net)
        #checkpoint["state_dict"] = self.state_dict()
        
        # recursive_print_dict(checkpoint["state_dict"])
        # equivariant models have to be in eval mode for saving the weights
        #if self.net.training:
        #    self.net.eval()
        #    checkpoint["state_dict"] = self.state_dict()
        #    self.net.train()


    def on_load_checkpoint(self, checkpoint):
        """This is called before load_state_dict()"""
        if not is_equivariant_model(self.net):
            # if the model is not equivariant nothing to do
            return
        
        if hasattr(self, "net"):
            if self.hparams.seed != checkpoint['seed']:
                # we can fix this by initializing a new model with the same seed
                raise ValueError(f"Seed {self.hparams.seed} is different from the one used to save the model {checkpoint['seed']}")
            # recreate the filters
            print("Creating filters")
            create_filters_network(self.net)
        else:
            # assume that basically everything is missing
            print("I think this does not exist")
            print(self)
            L.seed_everything(checkpoint['seed'], workers=True)


if __name__ == "__main__":
    _ = LitModule(None, None, None, None)
