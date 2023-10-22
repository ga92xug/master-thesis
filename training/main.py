from calendar import c
from typing import Dict, List, Tuple, Union
import numpy as np

np.set_printoptions(precision=3, linewidth=10000, suppress=True)
import hydra
from hydra.utils import instantiate, call
import os
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
import math
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics import MetricCollection
import sys
sys.path.append(os.getcwd()) # add current directory
from training.model_instantiate import get_model
from training import utils
from training.logger import Custom_Logger, SingletonInt
from training.wrapper_scheduler import Wrapper_Scheduler
os.environ['HYDRA_FULL_ERROR'] = '1'

class Experiment:
    def __init__(self, cfg: DictConfig):
        super(Experiment, self).__init__()
        # time
        self._time_limit = cfg.other.time_limit
        self._global_start_time = datetime.datetime.now()
        self._verbose = cfg.other.verbose
        self.is_nas = cfg.NAS.trial_index != -1
        self.wandb_run = utils.init_wandb(cfg)
        self.cfg = cfg
        self._iteration = SingletonInt("iteration", 0)
        self.epoch = SingletonInt("epoch", 0)
        self.global_step = SingletonInt("global_step", 0)  

        self.logger = Custom_Logger(
            cfg=cfg, 
            is_nas=self.is_nas, 
            ray=cfg.ray,
            max_epochs=cfg.training.epochs, 
            wandb_run=self.wandb_run,
            verbose=self._verbose,
            global_step=self.global_step,
            epoch=self.epoch,
        )
        self.logger.print_verbose_check(1, f"Stage 0: experiment starts at {self._global_start_time}")
        self.logger.print_verbose_check(1, f"{OmegaConf.to_yaml(cfg)}")
        # seed
        torch.manual_seed(cfg.other.seed)
        np.random.seed(cfg.other.seed)
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        

        # dataset
        self._dataloaders, normalize_weights = call(cfg.training.dataset, verbose=self._verbose)
        n_inputs = cfg.training.dataset.n_in_channels
        self.n_outputs = cfg.training.dataset.n_out_classes
        #self.n_outputs = 1 if self.n_outputs == 2 else self.n_outputs
        image_size = cfg.training.dataset.resolution
        self.distribution_shift = cfg.training.dataset.distribution_shift
        self.logger.print_verbose_check(1, "Stage 1: dataloaders built")

        # Metrics
        if self.n_outputs >= 2:
            self.train_metrics = {"acc": MulticlassAccuracy(self.n_outputs, average="micro").to(self.device)}
            self.valid_metrics = {"acc": MulticlassAccuracy(self.n_outputs, average="micro").to(self.device)}
            if normalize_weights is not None:
                # if we want to get the weighted acc we can pass a none list 
                self.train_metrics["acc_weighted"] = MulticlassAccuracy(self.n_outputs, average="macro").to(self.device)
                self.valid_metrics["acc_weighted"] = MulticlassAccuracy(self.n_outputs, average="macro").to(self.device)
                if isinstance(normalize_weights, list):
                    normalize_weights = torch.tensor(normalize_weights, dtype=torch.float32).to(self.device)
                else:
                    normalize_weights = None   
        else:
            assert normalize_weights is None, "Not implemented"    
            self.train_metrics = {"acc": BinaryAccuracy().to(self.device)}
            self.valid_metrics = {"acc": BinaryAccuracy().to(self.device)}


        self.train_metrics = MetricCollection(self.train_metrics)
        self.valid_metrics = MetricCollection(self.valid_metrics)
        
        # Loss function
        self._loss_function = torch.nn.CrossEntropyLoss(weight=normalize_weights)

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
        
        utils.give_wandb_name(self.model, self.wandb_run)
        self.logger.print_verbose_check(1, "Stage 2: model built")

        # optimizer
        self._optimizer = instantiate(cfg.training.optimizer, params=self.model.parameters())
        
        # paths
        self.output_path = utils.output_path(cfg.other.output_path)
        save_id = wandb.run.id if wandb.run.id is not None else self._global_start_time
        self.model_path = utils.backup_path(cfg.other.backup_model, self.output_path, save_id, verbose=self._verbose)

        # training configuration
        self.max_epochs = cfg.training.epochs
        self._eval_frequency = cfg.training.eval_frequency
        self.steps_per_epoch = cfg.training.steps_per_epoch
        self.valid_conf_matrix_frequency = cfg.other.valid_conf_matrix_frequency

        # adapt learning rate
        self.scheduler = Wrapper_Scheduler(cfg, self._optimizer, self._dataloaders, self.logger)
        
        # iteration is the number of batches seen
        self.train_n_batches_len = len(self._dataloaders["train"])      

        # early stopping
        self.early_stopping = None
        if cfg.training.earlystop.stop:
            self.early_stopping = utils.EarlyStopping(
                monitor=cfg.training.earlystop.monitor,
                mode=cfg.training.earlystop.mode,
                patience=cfg.training.earlystop.patience,
                min_delta=cfg.training.earlystop.min_delta,
                save_path=self.output_path + "best_model.pth" if self.model_path is None else self.model_path,
                verbose=self._verbose,
                baseline=cfg.training.earlystop.get("baseline", None), 
                store_in_memory=cfg.training.earlystop.store_in_memory,
                logger=self.logger,
            )
        self.logger.print_verbose_check(1, f"Stage 3: training starts {self._global_start_time}")
    

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
            x, t, _ = utils.get_out_dataloader(out_dataloader, self.device)            
            self.logger.print_verbose_check(3, f"\ttrain:{batch_idx}/{self.train_n_batches_len}\t\t{datetime.datetime.now()}")

            # compute prediction    
            y = self.model(x)
            # compute loss and accuracy
            n_samples += x.shape[0]
            loss = self._loss_function(y, t)
            metrics = self.train_metrics(y.detach(), t.detach()) 
            train_loss_epoch += loss.item() * x.shape[0]
            metrics["loss"] = loss.item()

            # log intermediate results
            self.logger.log(metrics, split="train", verbose=3)

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

        # log for full epoch
        end_time = datetime.datetime.now().timestamp()
        duration = end_time - start_time
        metric_full_epoch = self.train_metrics.compute()
        metric_full_epoch["loss"] = train_loss_epoch / n_samples
        metric_full_epoch["duration"] = duration
        self.logger.log(metric_full_epoch, split="train", verbose=0)
        
        # avoid wandb not logging duration bug
        self.global_step += 1
        self.train_metrics.reset()
        return

    def valid(self) -> Dict:
        confusion = self.valid_conf_matrix_frequency > 0 and \
            self.epoch % self.valid_conf_matrix_frequency == 0
        metrics, _, _ = self.inference("valid", confusion=confusion)
        
        # adapt learning rate
        self.scheduler.step_validation(metrics)
        return metrics


    @torch.no_grad()
    def inference(self, split, confusion=False) -> Tuple[Dict, float, float]:
        """
        Run inference.

        Returns:
        - metrics: dict of metrics
        - loss: float
        - duration: float
        """
        starttime = datetime.datetime.now().timestamp()
        self.model.eval()
        if confusion or self.distribution_shift:
            y_all = []
            t_all = []
            meta_data_all = []
        
        cumulative_loss = 0.
        n_samples = 0
        for _, out_dataloader in enumerate(self._dataloaders[split]):
            x, t, meta_data = utils.get_out_dataloader(out_dataloader, self.device)
 
            y = self.model(x)
            #y = y.squeeze() if y.shape[1] == 1 else y
            
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
            metrics = self._dataloaders[split].dataset.eval(torch.tensor(np.argmax(y_all, axis=1)), torch.tensor(t_all), torch.tensor(meta_data_all))[0]

        metrics["loss"] = loss
        metrics["duration"] = duration
        self.logger.log(metrics, split=split, verbose=0)

        if confusion:   
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=y_all, y_true=t_all, class_names=list(range(self.n_outputs)))})
            self.global_step += 1
        
        return metrics, loss, duration
    
    
    def iteration_over_epochs(self):
        """
        High level functionality to run the experiment. 
        Implements when to train, evaluate, plot, backup, etc.
        """
        self._iteration = 0
        
        while self.epoch < self.max_epochs and not self.time_limit_reached():
            # check if we are allowed to run
            utils.allowed_usage_time(self.cfg.other.gpu_time_limit)
            
            # train
            self.train()
            
            # validate
            if self._eval_frequency < 0 and self.epoch % (-self._eval_frequency) == 0:
                valid_metrics = self.valid()

                if self.early_stopping:
                    should_stop = self.early_stopping.should_stop(valid_metrics, self.model)
                    if should_stop:
                        self.model = self.early_stopping.restore_best_weights(self.model)
                        self.global_step += 1
                        break
            
            if self.cfg.other.backup_frequency < 0 and self.epoch % (-self.cfg.other.backup_frequency) == 0:
                self.backup()

            # adapt learning rate
            self.scheduler.step_epoch_end()

            self.epoch += 1
        
        # Training done, evaluate on test set
        self.backup()
        if self.cfg.other.should_test:
            self.global_step += 1
            self.inference("test", confusion=True)

    def time_limit_reached(self):
        if self._time_limit is not None and \
            (datetime.datetime.now().timestamp() - \
            self._global_start_time.timestamp()) / 60. \
            > self._time_limit:
            print(f"Time limit of {self._time_limit} minutes reached. Stopping training at epoch {self.epoch}.")
            return True
        return False

    def backup(self):
        if self.cfg.other.backup_model:
            torch.save(self.model.state_dict(), self.model_path)


def run_experiment(cfg: DictConfig) -> None:
    utils.debug_run(cfg)
    utils.allowed_usage_time(cfg.other.gpu_time_limit)
    exp = Experiment(cfg)
    exp.iteration_over_epochs()
    print(f"Experiment finished at {datetime.datetime.now()}")

    # clean up the experiment
    try:
        wandb.finish()
    except:
        pass
    del exp
    # Reset all instances
    SingletonInt.reset_all()

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def hydra_main_init(cfg: DictConfig) -> None:
    run_experiment(cfg)

def hydra_initialize_init(
        overrides: Union[Dict, List], 
        additional_overrides: Union[Dict, List] = {},
    ) -> None:
    assert type(overrides) == type(additional_overrides), "Overrides and additional_overrides must be of the same type"

    if isinstance(additional_overrides, dict):
        overrides = {**additional_overrides, **overrides}
    else:
        overrides = additional_overrides + overrides

    if isinstance(overrides, dict):
        overrides = [f"{key}={value if value != None else 'null'}" for key, value in overrides.items()]

    with hydra.initialize(config_path="conf", version_base="1.2"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    run_experiment(cfg)

if __name__ == "__main__":
    hydra_main_init()