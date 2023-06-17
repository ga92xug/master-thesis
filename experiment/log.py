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

            self.trial_index = self.cfg.NAS.trial_index
            self.prefix = f"{self.trial_index}_"


    def log(
            self, 
            to_log: dict, 
            step: int, 
            epoch: int,
        ):
        if self.is_nas:
            # Log to DB
            self.aggregate_and_log_db(to_log, self.trial_index, epoch)

            # Prefix log entries
            to_log = {self.prefix + key: value for key, value in to_log.items()}

        wandb.log(to_log, step=step)

    def aggregate_and_log_db(self, to_log: dict, trial_index: str, epoch: int):
        for key, sub_dict in to_log.items():
            if isinstance(sub_dict, dict):
                for sub_key, value in sub_dict.items():
                    new_key = f"{key}_{sub_key}"
                    if new_key in ['valid_acc', 'train_duration', 'valid_duration']:
                        # For 'valid_acc', 'train_duration', and 'valid_duration', only update the value if it's the last epoch
                        if epoch == self.max_epochs - 1:
                            self.trial_data[new_key] = value
                    else:
                        self.trial_data[new_key] = value
            else:
                self.trial_data[key] = sub_dict


        self.log_to_db(self.trial_data, trial_index)
        gflops = self.trial_data['GFLOPs']
        if gflops > self.cfg.NAS.max_gflops:
            raise ValueError(f"GFLOPs ({gflops}) is greater than the max ({self.cfg.NAS.max_gflops})")

    def log_to_db(self, to_log: dict, trial_index: int):
        for key, value in to_log.items():
            if isinstance(value, torch.Tensor):
                to_log[key] = value.item()
                

        keys_to_log = ['trial_index', 'GFLOPs', 'valid_acc', 'train_duration', 'valid_duration']
        values = [trial_index] + [to_log.get(key) for key in keys_to_log[1:]]

        query = f"INSERT OR REPLACE INTO run_metrics ({', '.join(keys_to_log)}) VALUES ({', '.join(['?'] * len(keys_to_log))})"
        self.cursor.execute(query, values)
        self.conn.commit()


    def connect_to_db(self):
        self.conn = sqlite3.connect(self.cfg.NAS.db_path)
        self.cursor = self.conn.cursor()





