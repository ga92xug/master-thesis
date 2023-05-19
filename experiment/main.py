import numpy as np
import math
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
import pprint
import sys

sys.path.append('../scaling-laws-ecnn') # add parent directory
from networks.util import get_param_count, cuda_memory_usage

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


#import e2cnn.nn as enn
import nn as enn

import pandas as pd
import argparse
import os
import datetime

# import plot_exps
import utils
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

def compute_confusion_matrix(predictions, targets, labels):
    if predictions.shape[1] > 1:
        predictions = predictions.argmax(axis=1)
    else:
        predictions = (predictions > 0.)
    
    conf_matrix = confusion_matrix(targets, predictions, labels=labels)
    return conf_matrix


def accuracy(predictions, targets):
    if predictions.shape[1] > 1:
        predictions = predictions.argmax(dim=1)
    else:
        predictions = (predictions > 0.)
    
    predictions = predictions.to(dtype=targets.dtype)
    accuracy = float((targets == predictions).sum()) / predictions.numel()
    return accuracy


class Experiment:
    def __init__(self, cfg: DictConfig):
        super(Experiment, self).__init__()
        # Wandb
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        # experiment name
        self.expname = utils.exp_name(cfg) if cfg.wandb.give_name else None
        run = wandb.init(project=cfg.wandb.project, config=wandb.config, mode=cfg.wandb.mode, \
                         name=self.expname, notes=cfg.wandb.notes, tags=cfg.wandb.tags)
        wandb.run.log_code(".")
        
        print(OmegaConf.to_yaml(cfg))
        self.cfg = cfg
        # seed
        torch.manual_seed(cfg.other.seed)
        np.random.seed(cfg.other.seed)
        
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        print("DEVICE:", self.device)

        # outpath
        # self.outpath = utils.out_path(cfg)
        # os.makedirs(self.outpath, exist_ok=True)
               
        # build the datasets and the train, validation and test loaders
        self._dataloaders, n_inputs, n_outputs = utils.build_dataloaders(cfg)
        print("Stage 1: datasets built")
        
        # Loss function
        if n_outputs == 2:
            n_outputs = 1
            self._loss_function = torch.nn.BCEWithLogitsLoss()
        else:
            self._loss_function = torch.nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        
        self.train_accuracy = MulticlassAccuracy(self.n_outputs).to(self.device) if self.n_outputs > 1 else BinaryAccuracy().to(self.device)

        # build the model
        self.model = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            image_size=cfg.dataset.resolution,
        ).to(self.device)
        if self.device != torch.device("cpu"):
            self.model = nn.DataParallel(self.model)
        if cfg.training.compile:
            self.model = torch.compile(self.model)
        print("Stage 2: model built")

        self._global_start_time = datetime.datetime.now()
        self._verbose = cfg.other.verbose
        total_param = get_param_count(self.model, in_mb=False, verbose=self._verbose)
        # assert total_param <= 4e7, "We don't want to train a model with more than 40M parameters!"
        wandb.log({"total_parameters": total_param}, step=0)
        if self._verbose > 1:
            print(f"Starting: {self._global_start_time}")
        
        # backup model parameters
        self.modelpath = utils.backup_path(cfg)
        if cfg.other.backup_model:
            os.makedirs(os.path.dirname(self.modelpath), exist_ok=True)
        print("modelpath", self.modelpath)

        # training configuration
        self.max_epochs = cfg.training.epochs
        self._eval_frequency = cfg.other.eval_frequency
        self.batch_size = cfg.training.batch_size
        self.accumulate = cfg.training.accumulate
        self.steps_per_epoch = cfg.training.steps_per_epoch
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
        
        # no hydra instantiation for optimizer since it is tricky to pass the model parameters
        #self._optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters(), 
        #                                          )
        if cfg.optimizer._target_ == "optimizer.build_optimizer_sfcnn":
            self._optimizer = hydra.utils.instantiate(cfg.optimizer, 
                                            params=self.model)
        else:
            self._optimizer = hydra.utils.instantiate(cfg.optimizer, 
                                            params=self.model.parameters())
        # self._optimizer = optimizer.build_optimizer(self.model, cfg)

        # adapt learning rate
        self._adapt_lr_type = cfg.training.adapt_lr
        if cfg.training.adapt_lr == "exponential":
            self._adapt_lr = self._lr_scheduler_exponential_decay
        elif cfg.training.adapt_lr == "validation":
            self._lr_scheduler = \
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer,
                    factor=self._lr_decay_factor,
                    patience=self._lr_decay_epoch,
                    verbose=self._verbose > 2,
                    eps=1e-8,
                )
            self._adapt_lr = self._lr_scheduler.step
            
        elif cfg.training.adapt_lr is not None:
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
        self.train_data_len = len(self._dataloaders["train"].dataset)
        self.actual_batch_size = self.batch_size * self.accumulate
        self.last_batch_size = self.train_data_len % self.actual_batch_size
        self.n_batches = self.train_data_len // self.actual_batch_size + (self.last_batch_size >= 0)        

    def log(self, accuracy, loss, split):
        """
        Plotting is currently not supported. We rely on wandb to plot the metrics.
        """
        return
        row = [self.seed, split, self._iteration, accuracy, loss]
        self.logs.loc[len(self.logs)] = row
    
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
        train_acc_epoch = 0
        n_samples = 0
        for batch_idx, (x, t) in enumerate(self._dataloaders["train"]):
            if self._verbose > 3:
                print(f"\ttrain:{batch_idx}/{self.train_n_batches_len}\t\t{datetime.datetime.now()}")
            
            n_samples += x.shape[0]

            x = x.to(self.device)
            t = t.to(self.device)
            y = self.model(x)
            #print("y", y.shape, y.dtype)
            #print("t", t.shape, t.dtype)
            #print("x", x.shape, x.dtype)
            loss = self._loss_function(y, t)
            acc = accuracy(y.detach(), t.detach())
                        
            train_loss_epoch += loss.item() * x.shape[0]
            train_acc_epoch += acc * x.shape[0]

            wandb.log({"train": {"loss": loss, "acc": acc}},\
                          step=self.global_step)
            if self._verbose > 2:
                    print(f"Epoch {self._epoch} | {epoch_iterations}/{self.n_batches};\
                           loss: {loss.item():.3f}; acc: {acc:.3f}")

            loss.backward(retain_graph=False)
            self.global_step += x.shape[0]

            # accumulate gradients
            if (batch_idx + 1) % self.accumulate == 0 or batch_idx == self.train_n_batches_len - 1:                
                self._optimizer.step()
                self._optimizer.zero_grad()
            
                if self._eval_frequency > 0 and self._iteration % self._eval_frequency == 0:
                    self.valid()
                
                self._iteration += 1
                epoch_iterations += 1
                
                if self.cfg.other.backup_frequency > 0 and self._iteration % self.cfg.other.backup_frequency == 0:
                    self.backup()

                if self.steps_per_epoch > 0 and epoch_iterations >= self.steps_per_epoch:
                    break

            if cuda_memory_usage(verbose=0) > 0.8:
                torch.cuda.empty_cache()

        # log the training loss, accuracy, and duration
        endtime = datetime.datetime.now().timestamp()
        duration = endtime - starttime
        wandb.log({"train": {"duration": duration}}, step=self.global_step)
        self.global_step += 1
        if self._verbose > 1:
            print(f"-"*100)
            print(f"TRAIN Epoch {self._epoch} lasted {duration:.3f} seconds")
            print(f'Accuracy: {train_acc_epoch / n_samples:.3f}; Loss: {(train_loss_epoch / n_samples):.3f}\n')
        return

    def test(self):
        if self._verbose > 0:
            print("\n")
            print("############################################ START TESTING ########################################")
        
        if self.cfg.training.earlystop:
            self.model.eval()
            self.model.load_state_dict(self.best_state_dict)
        
        acc, loss, duration, conf_matrix = self.evaluate("test", confusion=True)

        self.conf_matrix = conf_matrix
        
        if self._verbose > 0:
            np.set_printoptions(precision=4, suppress=True, threshold=1000000, linewidth=1000000)
            print(f"##### ExperimentClassification [{self.expname}]")
            print(f"##### TEST LOSS = {loss:.3f}")
            print(f"##### TEST ACCURACY = {acc:.3f}")
            print("###################################################################################################")
            print("# Confusion Matrix")
            print(conf_matrix)
            print("# Normalized Confusion Matrix")
            conf_matrix /= conf_matrix.sum(axis=1, keepdims=True)
            print(conf_matrix)
            print("###################################################################################################")
            print("\n")
    
    def valid(self):
        # during validation also evaluate test set. Don't do this
        if self.cfg.training.eval_test:
            acc, loss, duration = self.evaluate("test")
            if self._verbose > 1:
                self.print_results(acc, loss, duration, "TEST")

        # evaluate validation set
        acc, loss, duration = self.evaluate("valid")
        if self._verbose > 1:
            self.print_results(acc, loss, duration, "VALID")

        if self.cfg.training.earlystop or self._adapt_lr_type == "validation":
            # earlystop part
            if self._valid_metric == "accuracy":
                _last_valid_metric = acc
                if self.cfg.training.earlystop and _last_valid_metric > self.best_valid_accuracy:
                    self.best_valid_accuracy = _last_valid_metric
                    self.best_valid_iteration = self._epoch
                    self.best_state_dict = self.model.state_dict()
            elif self._valid_metric == "loss":
                _last_valid_metric = loss
                if self.cfg.training.earlystop and _last_valid_metric < self.best_valid_loss:
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
    def evaluate(self, split, log=True, confusion=False):
        starttime = datetime.datetime.now().timestamp()
        self.model.eval()
        if confusion:
            conf_matrix = np.zeros((self.n_outputs, self.n_outputs))
            y_test_all = []
            t_test_all = []
        
        cumulative_loss = 0.
        cumulative_acc = 0
        n_samples = 0
        for batch_idx, (x_test, t_test) in enumerate(self._dataloaders[split]):
            x_test = x_test.to(self.device)
            t_test = t_test.to(self.device)
            
            y_test = self.model(x_test)
            
            if confusion:
                y_test_all.append(y_test.detach().cpu().numpy())
                t_test_all.append(t_test.detach().cpu().numpy())

            n_samples += x_test.shape[0]
            cumulative_acc += accuracy(y_test, t_test) * x_test.shape[0]
            cumulative_loss += self._loss_function(y_test, t_test).item() * x_test.shape[0]
            
            del x_test
            del y_test
            del t_test
        
        acc = float(cumulative_acc / n_samples)
        loss = float(cumulative_loss / n_samples)        
        endtime = datetime.datetime.now().timestamp()
        duration = float(endtime - starttime)

        if log:
            wandb.log({f"{split}": {"loss": loss, "acc": acc, "duration": duration}}\
                          , step=self.global_step)

        if confusion:
            y_test_all = np.concatenate(y_test_all, axis=0)
            t_test_all = np.concatenate(t_test_all, axis=0)
            conf_matrix += compute_confusion_matrix(y_test_all, t_test_all, list(range(self.n_outputs)))
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=y_test_all, \
                            y_true=t_test_all, preds=None, class_names=list(range(self.n_outputs)))})
            return acc, loss, duration, conf_matrix
        else:
            return acc, loss, duration
    
    def print_results(self, acc, loss, duration, mode):
        print('-'*100)
        print(f'{mode} Epoch: {self._epoch} lasted {duration:.3f} seconds\nAccuracy: {acc:.3f}; Loss: {loss:.3f}\n')
    
    
    def run(self):
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
                if (datetime.datetime.now().timestamp() - self._global_start_time.timestamp()) / 60. > self._time_limit:
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

            self._epoch += 1
        
        # Training done, evaluate on test set
        if self._verbose > 1:
            print("###################################### Backup and Test #######################################")
        
        self.backup()
        if self.cfg.other.should_test:
            self.test()
        
        wandb.finish(exit_code=0)

    def _lr_scheduler_exponential_decay(self, verbose=False):
        #optimizer, epoch, epoch_start, init_lr, base_factor=.8, lr_decay_epoch=1, verbose=False):
        """
        Decay initial learning rate exponentially starting after epoch_start epochs
        The learning rate is multiplied with base_factor every lr_decay_epoch epochs
        """
        
        if self._lr_decay_schedule is not None:
            if self._epoch in self._lr_decay_schedule:
                self._lr_decay_schedule.remove(self._epoch)
                self._lr *= self._lr_decay_factor
            # count = len([e for e in self._lr_decay_schedule if e <= self._epoch])
            # lr = self._lr * (self._lr_decay_factor ** count)
        else:
            if self._epoch <= self._lr_decay_start:
                lr = self._lr
            else:
                lr = self._lr * (self._lr_decay_factor ** ((self._epoch - self._lr_decay_start) // self._lr_decay_epoch))
        if verbose:
            print('learning rate = {:6f}'.format(lr))
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return self._optimizer, lr
    
    def _lr_scheduler_valid_adaptive(self, verbose=False):
        """
        Decay initial learning rate exponentially by "_lr_decay_factor" starting after "_lr_decay_start" epochs
        The learning rate is multiplied with "_decay_factor" after the validation metric doesn't improve for "_lr_decay_epoch"
        """
        if self._epoch > self._lr_decay_start and self._epoch - max(self._last_adapt, self.best_valid_iteration) > 20:
            self._last_adapt = self._epoch
            self._lr_exp_steps += 1
        
        lr = self._lr * (self._lr_decay_factor ** self._lr_exp_steps)
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
    exp.run()
 
    
################################################################################
################################################################################


if __name__ == "__main__":
    run_experiment()
    
