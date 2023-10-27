import sqlite3
from sympy import Union
import torch
import wandb
from wandb.wandb_run import Run
from wandb.sdk.lib.disabled import RunDisabled
from typing import Union
from ray import train

from networks.util import flatten_dict

######################################################
# SingletonInt
######################################################

class SingletonInt:
    _instances = {}

    def __new__(cls, key, initial_value=0):
        if key not in cls._instances:
            cls._instances[key] = super(SingletonInt, cls).__new__(cls)
            cls._instances[key].value = initial_value
        return cls._instances[key]

    def __iadd__(self, other):
        self.value += other
        return self

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __le__(self, other):
        return self.value <= other

    def __ge__(self, other):
        return self.value >= other

    def __eq__(self, other):
        return self.value == other

    def __mod__(self, other):
        return self.value % other

    def __str__(self):
        return str(self.value)

    @classmethod
    def reset_all(cls):
        cls._instances = {}


class Custom_Logger():
    def __init__(
        self,
        cfg,
        wandb_run: Union[Run, RunDisabled],
        epoch: SingletonInt,
        global_step: SingletonInt,
        is_nas: bool = False,
        ray: bool = False,
        max_epochs: int = -1,  
        verbose: bool = False,   
    ):
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.is_nas = is_nas
        self.ray = ray
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.epoch = epoch
        self.global_step = global_step


        if self.is_nas:
            self.trial_data = {}
            self.connect_to_db()

            self.trial_index = self.cfg.NAS.trial_index
            self.prefix = f"{self.trial_index}_"


    def log(
            self, 
            to_log: dict, 
            split: str = None,
            verbose: int = 5,
        ):
        if self.verbose > verbose:
            if split is not None:
                self.print_results(to_log, split)
            else:
                print(to_log)

        if split is not None:
            to_log = {split: to_log}

        #print("to_log", to_log)
        if self.is_nas:
            # Log to DB
            self.aggregate_and_log_db(to_log, self.trial_index)

            # Prefix log entries
            #to_log = {self.prefix + key: value for key, value in to_log.items()}
        else:
            self.log2ray(to_log)
            self.wandb_run.log(to_log, step=self.global_step.value)
            # wandb logging bug if not increase global_step
            self.global_step += 1


    def aggregate_and_log_db(self, to_log: dict, trial_index: str):
        for key, sub_dict in to_log.items():
            if isinstance(sub_dict, dict):
                for sub_key, value in sub_dict.items():
                    new_key = f"{key}_{sub_key}"
                    if new_key in ['valid_acc_weighted', 'train_duration', 'valid_duration']:
                        # For 'valid_acc', 'train_duration', and 'valid_duration', only update the value if it's the last self.epoch
                        if self.epoch == self.max_epochs - 1:
                            self.trial_data[new_key] = value
                    else:
                        self.trial_data[new_key] = value
            else:
                self.trial_data[key] = sub_dict

        self.log_to_db(self.trial_data, trial_index)


    def log_to_db(self, to_log: dict, trial_index: int):
        for key, value in to_log.items():
            if isinstance(value, torch.Tensor):
                to_log[key] = value.item()
                
        keys_to_log = ['trial_index', 'GFLOPs', 'valid_acc_weighted', 'train_duration', 'valid_duration', "model_building_time"]
        values = [trial_index] + [to_log.get(key) for key in keys_to_log[1:]]

        query = f"INSERT OR REPLACE INTO run_metrics ({', '.join(keys_to_log)}) VALUES ({', '.join(['?'] * len(keys_to_log))})"
        self.cursor.execute(query, values)
        self.conn.commit()


    def connect_to_db(self):
        self.conn = sqlite3.connect(f"{self.cfg.NAS.db_path}")
        self.cursor = self.conn.cursor()


    def print_verbose_check(self, level: int, message: str) -> None:
        if self.verbose > level:
            print(message)


    def print_results(self, metrics, split):
        split = split.capitalize()
        duration = metrics.get("duration", 0)
        print('-'*80)
        print(f'{split} Epoch: {self.epoch} lasted {duration:.3f} seconds')
        metrics = ", ".join([f"{key}: {value:.3f}" for key, value in metrics.items() if key not in ["duration", "epoch"]])
        print(f'{metrics}')

    def log2ray(self, to_log: dict):
        if self.ray is None:
            return
        
        log_to_ray = to_python_obj(to_log)
        log_to_ray = flatten_dict(log_to_ray, separator=".")

        if log_to_ray.get(self.ray, None) is not None:
            log_to_ray = {self.ray: log_to_ray[self.ray]}
            print("log_to_ray", log_to_ray)
            train.report(log_to_ray)


def to_python_obj(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: to_python_obj(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_python_obj(element) for element in obj]
    else:
        return obj



