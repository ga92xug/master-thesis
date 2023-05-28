import wandb
from ax import Data
import pandas as pd
from ax.core.metric import Metric
from ax.core.outcome_constraint import ComparisonOp
from ax.core.types import Tuple
import numpy as np
import sqlite3
import time

# currently we use an evaluation function instead by we might need the 
# flexibility of a metric later
class WandbMetric(Metric):
    """
    A metric that fetches data from a wandb run.
    Implements a database cache to avoid repeated API calls.
    """

    def __init__(self, name: str, entity: str, project: str, 
                lower_is_better: bool, db_file: str = "data/wandb_cache.db"):
        super().__init__(name, lower_is_better=lower_is_better)
        self.project = project
        self.entity = entity
        self.db_file = db_file

    def connect_to_db(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wandb_run_metrics (
                run_id TEXT PRIMARY KEY,
                gflops REAL,
                acc_last_five TEXT
            )
        ''')
        conn.commit()
        return conn, cursor


    def fetch_trial_data(self, trial) -> Tuple:
        api = wandb.Api()
        trial_index = trial.index
        conn, cursor = self.connect_to_db()

        cursor.execute('SELECT * FROM wandb_run_metrics WHERE run_id = ?', (trial.index,))
        result = cursor.fetchone()

        if result is not None:
            gflops, mean_acc, train_duration, valid_duration = float(result[1]), float(result[2]), float(result[3]), float(result[4])
        else:
            successful_fetch = False
            while not successful_fetch:
                try:
                    run = api.run(f"{self.entity}/{self.project}/{trial.wandb_run_id}")
                    gflops = run.history(keys=['GFLOPs']).values[:,1][0]
                    acc = run.history(keys=['valid.acc']).values[:,1]
                    train_durations = run.history(keys=['train.duration']).values[:,1]
                    valid_durations = run.history(keys=['valid.duration']).values[:,1]
                    train_duration = np.median(train_durations)
                    valid_duration = np.median(valid_durations)
                    mean_acc = np.median(acc[-5:])
                    cursor.execute('INSERT INTO wandb_run_metrics VALUES (?, ?, ?, ?, ?)', 
                                   (trial.index, gflops, mean_acc, train_duration, valid_duration))
                    conn.commit()
                    successful_fetch = True
                except:
                    time.sleep(10)  # wait for 10 seconds before trying again

        data = {
            "gflops": gflops,
            "acc": mean_acc,
            "train_duration": train_duration,
            "valid_duration": valid_duration
        }
        return self._make_trial_data(trial_index, self.name, data[self.name])


    def to_json(self):
        return {
            "name": self.name,
            "entity": self.entity,
            "project": self.project,
            "db_file": self.db_file,
        }

    @classmethod
    def from_json(cls, json_repr):
        return cls(
            name=json_repr["name"],
            entity=json_repr["entity"],
            project=json_repr["project"],
            db_file=json_repr["db_file"],
        )



    def _make_trial_data(self, trial_index, metric_name, metric_value):
        """
        Convert fetched data into AX's TrialData format.
        """
        records = []
        records.append({
            "metric_name": metric_name,
            "mean": metric_value,
            "sem": np.NaN,  # Replace with standard error if available
            "trial_index": trial_index,
            "arm_name": str(trial_index),  # Replace with actual arm name
        })

        df = pd.DataFrame.from_records(records)
        df = df.astype({"metric_name": str, "mean": float, "trial_index": int, 
                        "arm_name": str})
        return Data(df=df)

    
    def is_available_while_running():
        """
        This would only be necessary for early stopping.
        Early stopping is not well supported in AX yet.
        We would have to change how we get mean_acc, and fetch the data from 
        wandb on every new step of the trial.
        https://ax.dev/tutorials/early_stopping/early_stopping.html
        """
        return False


    def get_metrics(self, trial):
        tried_fetching = False
        conn, cursor = self.connect_to_db()
        
        cursor.execute('SELECT * FROM wandb_run_metrics WHERE run_id = ?', (trial_data.index,))
        result = cursor.fetchone()
        while result is None and not tried_fetching:
            if result is not None:
                gflops, mean_acc, train_duration, valid_duration = float(result[1]), float(result[2]), float(result[3]), float(result[4])
                metrics = {
                    "metric_val_acc": mean_acc,
                    "metric_gflops": gflops,
                    "metric_train_duration": train_duration,
                    "metric_valid_duration": valid_duration
                }
                return metrics
            else:
                trial_data = self.fetch_trial_data(trial)
                tried_fetching = True