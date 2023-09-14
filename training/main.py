from email.mime import image
import sqlite3
import time
import numpy as np
np.set_printoptions(precision=3, linewidth=10000, suppress=True)
import hydra
import os
import datetime
from omegaconf import DictConfig, OmegaConf
import wandb
import math
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics import MetricCollection
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory
from training.model_instantiate import get_model
from networks.util import cuda_memory_usage
from training import utils
from training import log

os.environ['HYDRA_FULL_ERROR'] = '1'
#os.environ['TORCHDYNAMO_VERBOSE'] = '0'
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

from sklearn.metrics import balanced_accuracy_score, accuracy_score

class Experiment:
    def __init__(self, cfg: DictConfig):
        super(Experiment, self).__init__()
        self._verbose = cfg.other.verbose
        self._global_start_time = datetime.datetime.now()
        if self._verbose > 3:
            print(f"Starting: {self._global_start_time}")
            print(OmegaConf.to_yaml(cfg))
        # seed
        torch.manual_seed(cfg.other.seed)
        np.random.seed(cfg.other.seed)
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # Wandb
        self.cfg = cfg
        wandb_config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        
        # NAS runs have increasing trial index
        self.is_nas = cfg.NAS.trial_index != -1
        self.logger = log.Log(cfg=cfg, is_nas=self.is_nas, max_epochs=cfg.training.epochs)
        
        if not self.is_nas:
            # experiment name
            # normal training mode
            self.wandb_run = wandb.init(
                project=cfg.wandb.project, config=wandb_config, \
                            mode=cfg.wandb.mode, notes=cfg.wandb.notes, \
                            tags=cfg.wandb.tags)
            self.wandb_run.log_code(".")
            
        else:
            pass
               
        # dataset
        self._dataloaders, normalize_weights = hydra.utils.call(cfg.training.dataset)
        n_inputs = cfg.training.dataset.n_in_channels
        self.n_outputs = cfg.training.dataset.n_out_classes
        image_size = cfg.training.dataset.resolution
        self.distribution_shift = cfg.training.dataset.distribution_shift
        print("Stage 1: dataloaders built")
        
        # Loss function
        if normalize_weights is not None and isinstance(normalize_weights, list):
            normalize_weights = torch.tensor(normalize_weights, dtype=torch.float32).to(self.device)
            if self.n_outputs == 2:
                self._loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=normalize_weights)
            else:
                self._loss_function = torch.nn.CrossEntropyLoss(weight=normalize_weights)
        else:
            # normalize weight can be set to not a list in the dataloader to avoid the other loss
            if self.n_outputs == 2:
                self._loss_function = torch.nn.BCEWithLogitsLoss()
            else:
                self._loss_function = torch.nn.CrossEntropyLoss()
        
        # Metrics
        if self.n_outputs > 1:
            self.train_metrics = {"acc": MulticlassAccuracy(self.n_outputs, average="micro").to(self.device)}
            self.valid_metrics = {"acc": MulticlassAccuracy(self.n_outputs, average="micro").to(self.device)}
            if normalize_weights is not None:
                self.train_metrics["acc_weighted"] = MulticlassAccuracy(self.n_outputs, average="macro").to(self.device)
                self.valid_metrics["acc_weighted"] = MulticlassAccuracy(self.n_outputs, average="macro").to(self.device)

            self.train_metrics = MetricCollection(self.train_metrics)
            self.valid_metrics = MetricCollection(self.valid_metrics)
        else:
            assert normalize_weights is not None, "Not implemented"    
            self.train_metrics = BinaryAccuracy().to(self.device)
            self.valid_metrics = BinaryAccuracy().to(self.device)

        # model
        self.model, _ = get_model(
            cfg=cfg, 
            n_inputs=n_inputs, 
            n_outputs=self.n_outputs, 
            image_size=image_size,
            device=self.device,
            logger=self.logger,
            verbose=self._verbose,
        )
        
        if isinstance(cfg.wandb.give_name, str):
            self.wandb_run.name = cfg.wandb.give_name
        elif cfg.wandb.give_name:
            self.wandb_run.name = self.model.name
        print("Stage 2: model built")

        # optimizer
        self._optimizer = hydra.utils.instantiate(cfg.training.optimizer, 
                                            params=self.model.parameters())
        
        # outpath
        # self.outpath = utils.out_path(cfg)
        # os.makedirs(self.outpath, exist_ok=True)
        # backup model parameters
        if cfg.other.backup_model:
            self.modelpath = utils.backup_path(cfg)
            os.makedirs(os.path.dirname(self.modelpath), exist_ok=True)
            print("modelpath", self.modelpath)

        # training configuration
        self.max_epochs = cfg.training.epochs
        self._eval_frequency = cfg.other.eval_frequency
        self.steps_per_epoch = cfg.training.steps_per_epoch

        # adapt learning rate
        if "CosineAnnealingLR" in cfg.training.scheduler._target_:
            OmegaConf.set_readonly(cfg, False) 
            cfg.training.scheduler.T_max = len(self._dataloaders["train"]) * cfg.training.epochs
            OmegaConf.set_readonly(cfg, True) 
        self._lr_scheduler = hydra.utils.instantiate(cfg.training.scheduler, 
                                            optimizer=self._optimizer)
        print("Stage 3: optimizer built")
        self._adapt_lr_in_validation = "ReduceLROnPlateau" in cfg.training.scheduler._target_
        
        # iteration is the number of batches seen
        self._iteration = 0
        self._epoch = 0
        self.global_step = 0  
        self.train_n_batches_len = len(self._dataloaders["train"])      
        
        # time limit
        self._time_limit = cfg.other.time_limit
        self._global_start_time = datetime.datetime.now()
        print("Stage 4: training starts: " + str(self._global_start_time))

    
    def backup(self):
        if self.cfg.other.backup_model:
            torch.save(self.model.state_dict(), self.modelpath)
    
    def train(self):
        start_time = datetime.datetime.now().timestamp()
        self.model.train()
        self._optimizer.zero_grad()
        epoch_iterations = 0
        train_loss_epoch = 0
        n_samples = 0

        if self.cfg.wandb.watch:
            wandb.watch(self.model, self._loss_function, log="all", log_freq=10)

        # 1 epoch
        for batch_idx, out_dataloader in enumerate(self._dataloaders["train"]):
            
            if len(out_dataloader) == 2:
                x, t = out_dataloader
            elif len(out_dataloader) == 3:
                # domain shift
                x, t, _ = out_dataloader
            else:
                raise ValueError("Dataloader should return 2 or 3 values")
            

            if self._verbose > 3:
                print(f"\ttrain:{batch_idx}/{self.train_n_batches_len}\t\t{datetime.datetime.now()}")
            
            # compute prediction
            x = x.to(self.device)
            t = t.to(self.device)
            y = self.model(x)

            # compute loss and accuracy
            n_samples += x.shape[0]
            loss = self._loss_function(y, t)
            metrics = self.train_metrics(y.detach(), t.detach()) 
            train_loss_epoch += loss.item() * x.shape[0]
            self.logger.log({"train": {"loss": loss} | metrics}, step=self.global_step, epoch=self._epoch)
            if self._verbose > 2:
                print(f"Epoch {self._epoch}: loss: {loss.item():.3f},", ', '.join([f'{key}: {value:.3f}' for key, value in metrics.items()]))

            # loss
            loss = loss / self.cfg.training.accumulate
            loss.backward()
            
            # accumulate gradients
            if (batch_idx + 1) % self.cfg.training.accumulate == 0 or \
                 batch_idx == self.train_n_batches_len - 1:                
                self._optimizer.step()
                self._optimizer.zero_grad()
                self._iteration += 1
                epoch_iterations += 1

                # validation more than once per epoch
                if self._eval_frequency > 0 and self._iteration % self._eval_frequency == 0:
                    self.valid()
                
                # short training for testing
                if self.steps_per_epoch > 0 and epoch_iterations >= self.steps_per_epoch:
                    break
            
            self.global_step += x.shape[0]
            # if cuda_memory_usage(verbose=0) > 0.8:
            #     torch.cuda.empty_cache()

        # log and print
        end_time = datetime.datetime.now().timestamp()
        duration = end_time - start_time
        self.logger.log({"train": {"duration": duration}}, step=self.global_step, epoch=self._epoch)
        utils.print_results(self.train_metrics.compute(), train_loss_epoch / n_samples, duration, "TRAIN", self._epoch, self._verbose)
        
        # avoid wandb not logging duration bug
        self.global_step += 1
        self.train_metrics.reset()
        return

    def valid(self):
        metrics, _, _ = self.inference("valid", confusion=False)
        
        # adapt learning rate
        if self._adapt_lr_in_validation:
            self._lr_scheduler.step(metrics["acc"])


    @torch.no_grad()
    def inference(self, split, log=True, confusion=False):
        starttime = datetime.datetime.now().timestamp()
        self.model.eval()
        if confusion or self.distribution_shift:
            y_all = []
            t_all = []
            meta_data_all = []
        
        cumulative_loss = 0.
        n_samples = 0
        for _, out_dataloader in enumerate(self._dataloaders[split]):
            if len(out_dataloader) == 2:
                x, t = out_dataloader
                meta_data = None
            elif len(out_dataloader) == 3:
                # domain shift
                x, t, meta_data = out_dataloader
            else:
                raise ValueError("Dataloader should return 2 or 3 values")

            x = x.to(self.device)
            t = t.to(self.device)
            y = self.model(x)
            
            if confusion or self.distribution_shift:
                y_all.append(y.detach().cpu().numpy())
                t_all.append(t.detach().cpu().numpy())
                meta_data_all.append(meta_data)

            n_samples += x.shape[0]
            self.valid_metrics(y, t)
            cumulative_loss += self._loss_function(y, t).item() * x.shape[0]
            
            del x
            del y
            del t
            del meta_data

        # log
        if confusion or self.distribution_shift:
            y_all = np.concatenate(y_all, axis=0)
            t_all = np.concatenate(t_all, axis=0)

        loss = float(cumulative_loss / n_samples)        
        endtime = datetime.datetime.now().timestamp()
        duration = float(endtime - starttime)
        if not self.distribution_shift:
            metrics = self.valid_metrics.compute()
            self.valid_metrics.reset()
        else:
            meta_data_all = np.concatenate(meta_data_all, axis=0)
            y_all = np.argmax(y_all, axis=1)
            metrics = self._dataloaders[split].dataset.eval(torch.tensor(y_all), torch.tensor(t_all), torch.tensor(meta_data_all))[0]

        self.logger.log(
            {f"{split}": {"loss": loss, "duration": duration} | metrics}, 
            step=self.global_step, epoch=self._epoch)

        if confusion:
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=y_all, y_true=t_all, class_names=list(range(self.n_outputs)))})
        
        # print
        split = split.upper()
        utils.print_results(metrics, loss, duration, split, self._epoch, self._verbose)
        return metrics, loss, duration
    
    
    def iteration_over_epochs(self):
        """
        High level functionality to run the experiment. 
        Implements when to train, evaluate, plot, backup, etc.
        """
        self._iteration = 0
        
        while self._epoch < self.max_epochs and not self.time_limit_reached():
            # check if we are allowed to run
            utils.allowed_usage_time(self.cfg.other.gpu_time_limit)
            
            # train
            self.train()
            
            # validate
            if self._eval_frequency < 0 and self._epoch % (-self._eval_frequency) == 0:
                self.valid()
            
            if self.cfg.other.backup_frequency < 0 and self._epoch % (-self.cfg.other.backup_frequency) == 0:
                self.backup()

            # adapt learning rate
            if not self._adapt_lr_in_validation and self._lr_scheduler is not None:
                self._lr_scheduler.step()

            self._epoch += 1
        
        # Training done, evaluate on test set
        self.backup()
        if self.cfg.other.should_test:
            self.inference("test", confusion=True)
        
        if not self.is_nas:
            wandb.finish()

    def time_limit_reached(self):
        if self._time_limit is not None and \
            (datetime.datetime.now().timestamp() - \
            self._global_start_time.timestamp()) / 60. \
            > self._time_limit:
            print(f"Time limit of {self._time_limit} minutes reached. Stopping training at epoch {self._epoch}.")
            return True
        return False

    

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def run_experiment(cfg: DictConfig) -> None:
    # check if we are allowed to run
    utils.allowed_usage_time(cfg.other.gpu_time_limit)
    exp = Experiment(cfg)
    exp.iteration_over_epochs()


def run_experiment_from_config(cfg: DictConfig) -> None:
    # check if we are allowed to run
    utils.allowed_usage_time(cfg.other.gpu_time_limit)
    exp = Experiment(cfg)
    exp.iteration_over_epochs()


if __name__ == "__main__":
    run_experiment()