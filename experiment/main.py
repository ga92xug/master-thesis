from tabnanny import verbose
import numpy as np
import hydra
import os
import datetime
from omegaconf import DictConfig, OmegaConf
import wandb
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchmetrics.classification import (
    BinaryAccuracy, 
    MulticlassAccuracy,
)
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

from networks.util import get_param_count, cuda_memory_usage, get_gflops
import utils
import log
import optimizer
#import optimizers_L1L2

from sklearn.metrics import confusion_matrix

import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, linewidth=10000, suppress=True)

#os.environ['HYDRA_FULL_ERROR'] = '1'
#os.environ['TORCHDYNAMO_VERBOSE'] = '0'
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True


class Experiment:
    def __init__(self, cfg: DictConfig):
        super(Experiment, self).__init__()
        self._verbose = cfg.other.verbose
        self._global_start_time = datetime.datetime.now()
        if self._verbose > 1:
            print(f"Starting: {self._global_start_time}")
        # seed
        torch.manual_seed(cfg.other.seed)
        np.random.seed(cfg.other.seed)
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # Wandb
        print(OmegaConf.to_yaml(cfg))
        self.cfg = cfg
        wandb_config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        
        # NAS runs have increasing trial index
        self.is_nas = cfg.NAS.trial_index != -1
        self.logger = log.Log(cfg=cfg, is_nas=self.is_nas, max_epochs=cfg.other.max_epochs)
        
        if not self.is_nas:
            # experiment name
            self.expname = utils.exp_name(cfg) if cfg.wandb.give_name else None

            # normal training mode
            self.wandb_run = wandb.init(
                project=cfg.wandb.project, config=wandb_config, \
                            mode=cfg.wandb.mode, name=self.expname, \
                            notes=cfg.wandb.notes, tags=cfg.wandb.tags)
            self.wandb_run.log_code(".")
            
        else:
            # console logging is a problem when running 2 wandb runs in parallel
            # so we disable it https://github.com/wandb/wandb/issues/4872
            os.environ['WANDB_CONSOLE']="off"
            os.environ['WANDB_DISABLE_SERVICE']='true'
            os.environ["WANDB_SILENT"] = "true"
            # during NAS we reinit
            self.wandb_run = wandb.init(
                id = cfg.wandb.run_id, 
                resume = "allow", 
                project = cfg.wandb.project, 
                entity = cfg.wandb.entity,
                mode = cfg.wandb.mode,
                notes = cfg.wandb.notes,
                tags = cfg.wandb.tags
            )
               
        # dataset
        self._dataloaders, n_inputs, self.n_outputs = utils.build_dataloaders(cfg)
        print("Stage 1: datasets built")
        
        # Loss function
        if self.n_outputs == 2:
            self._loss_function = torch.nn.BCEWithLogitsLoss()
        else:
            self._loss_function = torch.nn.CrossEntropyLoss()
        
        # Metrics
        if self.n_outputs > 1:
            self.train_accuracy = MulticlassAccuracy(self.n_outputs, average="micro").to(self.device)
            self.valid_accuracy = MulticlassAccuracy(self.n_outputs, average="micro").to(self.device)
        else:
            self.train_accuracy = BinaryAccuracy().to(self.device)
            self.valid_accuracy = BinaryAccuracy().to(self.device)

        # model
        self.model = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=self.n_outputs,
            image_size=cfg.dataset.resolution,
        ).to(self.device)
        if cfg.training.compile:
            self.model = torch.compile(self.model)
        print("Stage 2: model built")

        # optimizer
        self._optimizer = hydra.utils.instantiate(cfg.optimizer, 
                                            params=self.model.parameters())

        # total parameters and GFLOPs
        total_param = get_param_count(self.model, in_mb=False, \
                                      verbose=self._verbose)
        gflops = get_gflops(self.model, cfg, n_inputs, verbose=self._verbose)
        self.logger.log({"total_parameters": total_param,
                   "GFLOPs": gflops}, step=0, epoch=0)
        # assert total_param <= 4e7, "We don't want to train a model with more than 40M parameters!"
        
        # outpath
        # self.outpath = utils.out_path(cfg)
        # os.makedirs(self.outpath, exist_ok=True)
        # backup model parameters
        self.modelpath = utils.backup_path(cfg)
        if cfg.other.backup_model:
            os.makedirs(os.path.dirname(self.modelpath), exist_ok=True)
        print("modelpath", self.modelpath)

        # training configuration
        self.max_epochs = cfg.training.epochs
        self._eval_frequency = cfg.other.eval_frequency
        self.steps_per_epoch = cfg.training.steps_per_epoch

        # learning rate
        self._lr = cfg.optimizer.lr
        self._lr_decay_start = cfg.training.lr_decay_start
        self._lr_decay_factor = cfg.training.lr_decay_factor
        self._lr_decay_epoch = cfg.training.lr_decay_epoch
        self._lr_decay_schedule = cfg.training.lr_decay_schedule
        print("lr_decay_schedule: ", cfg.training.lr_decay_schedule)
        if cfg.training.lr_decay_schedule is not None:
            print("if statement lr_decay_schedule: ", cfg.training.lr_decay_schedule is not None)
            self._lr_decay_epoch = None
            self._lr_decay_start = None
        
        self._lr_exp_steps = 0

        # adapt learning rate
        self._adapt_lr_type = cfg.training.adapt_lr
        if self._adapt_lr_type == "exponential":
            self._lr_scheduler = \
                MultiStepLR(
                    self._optimizer,
                    milestones=[self._lr_decay_start],
                    gamma=self._lr_decay_factor,
                )
            # self._adapt_lr = self._lr_scheduler_exponential_decay
        elif self._adapt_lr_type == "validation":
            self._lr_scheduler = \
                ReduceLROnPlateau(
                    self._optimizer,
                    factor=self._lr_decay_factor,
                    patience=self._lr_decay_epoch,
                    verbose=self._verbose > 2,
                    eps=1e-8,
                )
            self._adapt_lr = self._lr_scheduler.step
            
        elif self._adapt_lr_type is not None:
            raise ValueError()
        else:
            self._adapt_lr = None
        
        # iteration is the number of batches seen
        self._iteration = 0
        self._epoch = 0
        self.global_step = 0        
        
        assert cfg.training.valid_metric in ["loss", "accuracy"]
        self._valid_metric = cfg.training.valid_metric

        self._last_valid_metric = 1e+20
        self.best_valid_iteration = 0
        self.best_valid_loss = 1e+20
        self.best_valid_accuracy = 0
        self.best_state_dict = self.model.state_dict()
        
        self._time_limit = cfg.other.time_limit
        self._global_start_time = datetime.datetime.now()

        # training statistics
        self.train_n_batches_len = len(self._dataloaders["train"])
    
    def backup(self):
        if self.cfg.other.backup_model:
            torch.save(self.best_state_dict, self.modelpath)
    
    def train(self):
        starttime = datetime.datetime.now().timestamp()

        if self.cfg.wandb.watch:
            # Tell wandb to watch what the model gets up to: gradients, weights, and more!
            wandb.watch(self.model, self._loss_function, log="all", log_freq=10)

        self.model.train()
        self._optimizer.zero_grad()
        epoch_iterations = 0
        train_loss_epoch = 0
        n_samples = 0

        # 1 epoch
        for batch_idx, (x, t) in enumerate(self._dataloaders["train"]):
            if self._verbose > 3:
                print(f"\ttrain:{batch_idx}/{self.train_n_batches_len}\t\t{datetime.datetime.now()}")
            
            # compute prediction
            x = x.to(self.device)
            t = t.to(self.device)
            y = self.model(x)

            # compute loss and accuracy
            n_samples += x.shape[0]
            loss = self._loss_function(y, t)
            acc = self.train_accuracy(y.detach(), t.detach())    
            train_loss_epoch += loss.item() * x.shape[0]
            wandb.log({"train": {"loss": loss, "acc": acc}}, step=self.global_step)
            if self._verbose > 2:
                print(f"Epoch {self._epoch} \
                        loss: {loss.item():.3f}; acc: {acc:.3f}")

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
        endtime = datetime.datetime.now().timestamp()
        duration = endtime - starttime
        self.logger.log({"train": {"duration": duration}}, step=self.global_step, epoch=self._epoch)
        utils.print_results(self.train_accuracy.compute(), train_loss_epoch / n_samples, duration, "TRAIN", self._epoch, self._verbose)
        
        # avoid wandb not logging duration bug
        self.global_step += 1
        self.train_accuracy.reset()
        return

    def test(self):
        if self.cfg.training.earlystop:
            self.model.load_state_dict(self.best_state_dict)
        
        self.inference("test", confusion=True)
    
    def valid(self):
        acc, loss, duration = self.inference("valid")
        utils.print_results(acc, loss, duration, "VALID", self._epoch, self._verbose)

        if self.cfg.training.earlystop or self._adapt_lr_type == "validation":
            # earlystop part
            if self._valid_metric == "accuracy":
                _last_valid_metric = acc
                if self.cfg.training.earlystop and \
                    _last_valid_metric > self.best_valid_accuracy:
                    
                    self.best_valid_accuracy = _last_valid_metric
                    self.best_valid_iteration = self._epoch
                    self.best_state_dict = self.model.state_dict()
            elif self._valid_metric == "loss":
                _last_valid_metric = loss
                if self.cfg.training.earlystop and \
                    _last_valid_metric < self.best_valid_loss:

                    self.best_valid_loss = _last_valid_metric
                    self.best_valid_iteration = self._epoch
                    self.best_state_dict = self.model.state_dict()
            else:
                raise ValueError(self._valid_metric)
            
            # adapt learning rate
            if self._adapt_lr_type == "validation" and self._adapt_lr is not None:
                self._adapt_lr(_last_valid_metric)

            self.best_valid_loss = min(loss, self.best_valid_loss)
            self.best_valid_accuracy = max(acc, self.best_valid_accuracy)

    @torch.no_grad()
    def inference(self, split, log=True, confusion=False):
        starttime = datetime.datetime.now().timestamp()
        self.model.eval()
        if confusion:
            y_test_all = []
            t_test_all = []
        
        cumulative_loss = 0.
        n_samples = 0
        for _, (x_test, t_test) in enumerate(self._dataloaders[split]):
            x_test = x_test.to(self.device)
            t_test = t_test.to(self.device)
            
            y_test = self.model(x_test)
            
            if confusion:
                y_test_all.append(y_test.detach().cpu().numpy())
                t_test_all.append(t_test.detach().cpu().numpy())

            n_samples += x_test.shape[0]
            self.valid_accuracy(y_test, t_test)
            cumulative_loss += self._loss_function(y_test, t_test).item() * x_test.shape[0]
            
            del x_test
            del y_test
            del t_test
        
        # log
        acc = self.valid_accuracy.compute()
        self.valid_accuracy.reset()
        loss = float(cumulative_loss / n_samples)        
        endtime = datetime.datetime.now().timestamp()
        duration = float(endtime - starttime)
        wandb.log({f"{split}": {"loss": loss, "acc": acc, \
                            "duration": duration}}, step=self.global_step)
        if confusion:
            y_test_all = np.concatenate(y_test_all, axis=0)
            t_test_all = np.concatenate(t_test_all, axis=0)
            wandb.log({"confusion_matrix": \
                        wandb.plot.confusion_matrix(probs=y_test_all,
                        y_true=t_test_all, \
                        class_names=list(range(self.n_outputs)))})
        return acc, loss, duration
    
    
    def iteration_over_epochs(self):
        """
        High level functionality to run the experiment. 
        Implements when to train, evaluate, plot, backup, etc.
        """
        self._iteration = 0
        
        while self._epoch < self.max_epochs:
            # check if we are allowed to run
            if self.cfg.other.gpu_time_limit:
                utils.allowed_usage_time()
            
            if self._time_limit is not None:
                if (datetime.datetime.now().timestamp() - \
                    self._global_start_time.timestamp()) / 60. \
                    > self._time_limit:
                    print(f"Time limit of {self._time_limit} minutes reached. Stopping training at epoch {self._epoch}.")
                    print(f"Best validation accuracy: {self.best_valid_accuracy:.3f}")
                    print(f"Best validation loss: {self.best_valid_loss:.3f}")
                    print(f"Best validation iteration: {self.best_valid_iteration}")
                    break
            
            if self._adapt_lr is not None and self._adapt_lr_type != "validation":
                self._adapt_lr()
            
            self.train()
            
            if self._eval_frequency < 0 and self._epoch % (-self._eval_frequency) == 0:
                self.valid()
            
            if self.cfg.other.backup_frequency < 0 and self._epoch % (-self.cfg.other.backup_frequency) == 0:
                self.backup()

            # adapt learning rate


            self._epoch += 1
        
        # Training done, evaluate on test set
        self.backup()
        if self.cfg.other.should_test:
            self.test()
        
        wandb.finish(exit_code=0)

    def _lr_scheduler_exponential_decay(self, verbose=False):
        """
        Decay initial learning rate exponentially starting after epoch_start epochs
        The learning rate is multiplied with base_factor every lr_decay_epoch epochs
        """
        
        if self._lr_decay_schedule is not None:
            if self._epoch in self._lr_decay_schedule:
                self._lr_decay_schedule.remove(self._epoch)
                self._lr *= self._lr_decay_factor
        else:
            if self._epoch <= self._lr_decay_start:
                lr = self._lr
            else:
                lr = self._lr * (self._lr_decay_factor ** \
                                 ((self._epoch - self._lr_decay_start) // \
                                  self._lr_decay_epoch))
        if verbose:
            print('learning rate = {:6f}'.format(lr))
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return self._optimizer, lr
    
    

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def run_experiment(cfg: DictConfig) -> None:
    # check if we are allowed to run
    if cfg.other.gpu_time_limit:
        utils.allowed_usage_time()
    exp = Experiment(cfg)
    exp.iteration_over_epochs()
 

if __name__ == "__main__":
    run_experiment()
    
