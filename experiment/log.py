import sqlite3
import torch
import wandb

class Log():
    def __init__(
            self,
            cfg,
            is_nas: bool = False,
            max_epochs: int = -1,
    ):
        self.cfg = cfg
        self.is_nas = is_nas
        self.max_epochs = max_epochs

        if self.is_nas:
            self.trial_data = {}
            self.connect_to_db()


    def log(self, to_log: dict, step: int, epoch: int):
        if self.is_nas:
            trial_index = self.cfg.NAS.trial_index
            prefix = f"{trial_index}_"

            self.aggregate_and_log_db(to_log, trial_index, epoch)

            # Prefix log entries
            to_log = {prefix + key: value for key, value in to_log.items()}

        wandb.log(to_log, step)

    def aggregate_and_log_db(self, to_log: dict, trial_index: str, epoch: int):
        if trial_index not in self.trial_data:
            # Initialize dict for trial if it doesn't exist
            self.trial_data[trial_index] = {}

        for key, sub_dict in to_log.items():
            if isinstance(sub_dict, dict):
                for sub_key, value in sub_dict.items():
                    new_key = f"{key}_{sub_key}"
                    if new_key in ['valid_acc', 'train_duration', 'valid_duration']:
                        # For 'valid_acc', 'train_duration', and 'valid_duration', only update the value if it's the last epoch
                        if epoch == self.max_epochs - 1 and new_key not in self.trial_data[trial_index]:
                            self.trial_data[trial_index][new_key] = value
                    else:
                        self.trial_data[trial_index][new_key] = value
            else:
                self.trial_data[trial_index][key] = sub_dict

        # Check if all keys are populated
        if all(key in self.trial_data[trial_index] for key in ['GFLOPs', 'valid_acc', 'train_duration', 'valid_duration']):
            # If all keys are populated, write to DB
            self.log_to_db(self.trial_data[trial_index], trial_index)
        
        elif self.trial_data[trial_index]['GFLOPs'] > self.cfg.nas.max_gflops:
            # If GFLOPs is greater than the max, write to DB
            self.log_to_db(self.trial_data[trial_index], trial_index)
            raise ValueError(f"GFLOPs ({self.trial_data[trial_index]['GFLOPs']}) is greater than the max ({self.cfg.nas.max_gflops})")

    def log_to_db(self, to_log: dict, trial_index: int):
        keys_to_log = ['trial_index', 'GFLOPs', 'valid_acc', 'train_duration', 'valid_duration']
        values = [trial_index] + [to_log.get(key) for key in keys_to_log[1:]]

        query = f"INSERT INTO run_metrics ({', '.join(keys_to_log)}) VALUES ({', '.join(['?'] * len(keys_to_log))})"
        self.cursor.execute(query, values)
        self.conn.commit()


    def connect_to_db(self):
        self.conn = sqlite3.connect(self.cfg.NAS.db_path)
        self.cursor = self.conn.cursor()





