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
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory
from experiment.model_instantiate import get_model
from networks.util import cuda_memory_usage
from experiment import utils
from experiment import log

from experiment.datasets.mnist import data_loader_mnist_rot
from experiment.datasets.mnist_fliprot import data_loader_mnist_fliprot
from experiment.datasets.mnist12k import data_loader_mnist12k
from experiment.datasets.cifar import data_loader_cifar10
from experiment.datasets.cifar100 import data_loader_cifar100
# from experiment.datasets.STL10 import data_loader_stl10
# from experiment.datasets.STL10 import data_loader_stl10frac
from experiment.datasets.imagenette import data_loader_imagenette
from experiment.datasets.Galaxy10_DECals import data_loader_Galaxy10_DECals


os.environ['HYDRA_FULL_ERROR'] = '1'
#os.environ['TORCHDYNAMO_VERBOSE'] = '0'
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

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
            self.expname = utils.exp_name(cfg) if cfg.wandb.give_name else None

            # normal training mode
            self.wandb_run = wandb.init(
                project=cfg.wandb.project, config=wandb_config, \
                            mode=cfg.wandb.mode, name=self.expname, \
                            notes=cfg.wandb.notes, tags=cfg.wandb.tags)
            self.wandb_run.log_code(".")
            
        else:
            pass
            # console logging is a problem when running 2 wandb runs in parallel
            # so we disable it https://github.com/wandb/wandb/issues/4872
            # os.environ['WANDB_CONSOLE']="off"
            # os.environ['WANDB_DISABLE_SERVICE']='true'
            # os.environ["WANDB_SILENT"] = "true"
            # # during NAS we reinit
            # self.wandb_run = wandb.init(
            #     id = cfg.wandb.run_id, 
            #     resume = "allow", 
            #     project = cfg.wandb.project, 
            #     entity = cfg.wandb.entity,
            #     mode = cfg.wandb.mode,
            #     notes = cfg.wandb.notes,
            #     tags = cfg.wandb.tags
            # )
               
        # dataset
        try:
            self._dataloaders, n_inputs, self.n_outputs, normalize_weights  = hydra.utils.call(cfg.training.dataset)
            #self._dataloaders, n_inputs, self.n_outputs = utils.build_dataloaders(cfg)
            
        except Exception as e:
            print(e)
            print("\nFailed --------------------------")
            self._dataloaders, n_inputs, self.n_outputs = utils.build_dataloaders(cfg)
            print("\nFailed --------------------------")

        print("Stage 1: dataloaders built")
        
        # Loss function
        if normalize_weights is not None:
            normalize_weights = torch.tensor(normalize_weights).to(self.device)
            if self.n_outputs == 2:
                self._loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=normalize_weights)
            else:
                self._loss_function = torch.nn.CrossEntropyLoss(weight=normalize_weights)
        else:
            if self.n_outputs == 2:
                self._loss_function = torch.nn.BCEWithLogitsLoss()
            else:
                self._loss_function = torch.nn.CrossEntropyLoss()
        
        # Metrics
        if self.n_outputs > 1:
            if normalize_weights is not None:
                average = "weighted"
            else:
                average = "micro"
            self.train_accuracy = MulticlassAccuracy(self.n_outputs, average=average).to(self.device)
            self.valid_accuracy = MulticlassAccuracy(self.n_outputs, average=average).to(self.device)
        else:
            assert normalize_weights is not None, "Not implemented"    
            self.train_accuracy = BinaryAccuracy().to(self.device)
            self.valid_accuracy = BinaryAccuracy().to(self.device)

        # model
        self.model, stats = get_model(
            cfg=cfg, 
            n_inputs=n_inputs, 
            n_outputs=self.n_outputs, 
            image_size=cfg.training.dataset.resolution,
            device=self.device,
            logger=self.logger,
            verbose=self._verbose,
        )
        try:
            self.wandb_run.name = self.model.name
        except:
            # not every model implements name yet
            pass
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
            self.logger.log({"train": {"loss": loss, "acc": acc}}, step=self.global_step, epoch=self._epoch)
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
        end_time = datetime.datetime.now().timestamp()
        duration = end_time - start_time
        self.logger.log({"train": {"duration": duration}}, step=self.global_step, epoch=self._epoch)
        utils.print_results(self.train_accuracy.compute(), train_loss_epoch / n_samples, duration, "TRAIN", self._epoch, self._verbose)
        
        # avoid wandb not logging duration bug
        self.global_step += 1
        self.train_accuracy.reset()
        return

    def test(self):
        self.inference("test", confusion=True)
    
    def valid(self):
        acc, loss, duration = self.inference("valid")
        utils.print_results(acc, loss, duration, "VALID", self._epoch, self._verbose)
        
        # adapt learning rate
        if self._adapt_lr_in_validation:
            self._lr_scheduler.step(acc)


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
        self.logger.log(
            {f"{split}": {"loss": loss, "acc": acc, "duration": duration}}, 
            step=self.global_step, epoch=self._epoch)
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
            self.test()
        
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