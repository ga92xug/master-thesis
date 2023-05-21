import wandb
from ax import Data
import pandas as pd
from ax.core.metric import Metric
from ax.core.outcome_constraint import ComparisonOp
from ax.core.types import Tuple
import numpy as np
import sqlite3

class WandbMetric(Metric):
    """
    This class is used to fetch data from wandb.
    The data is stored in a local database to avoid fetching the same data
    multiple times.
    For early stopping change that.
    """
    def __init__(self, name: str, entity: str, project: str, 
                 db_file: str = "data/wandb_cache.db"):
        super().__init__(name)
        self.api = wandb.Api()
        self.project = project
        self.entity = entity
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS wandb_run_metrics (
                run_id TEXT PRIMARY KEY,
                gflops REAL,
                acc_last_five TEXT
            )
        ''')
        self.conn.commit()

    def fetch_trial_data(self, trial) -> Tuple:
        """
        Fetch data for a specific trial.
        """
        # trial_index = trial.index
        trial_index = trial
        run_id = trial_index  # Assuming the run id follows this format
        self.cursor.execute('SELECT * FROM wandb_run_metrics WHERE run_id = ?', 
                            (run_id,))
        result = self.cursor.fetchone()
        print("result", result)

        if result is not None:
            gflops, mean_acc = float(result[1]), float(result[2])
            print("gflops", gflops)
            print("mean_acc", mean_acc)
        else:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            # Fetch metrics
            gflops = run.history(keys=['GFLOPs']).values[:,1][0]
            acc = run.history(keys=['valid.acc']).values[:,1]
            # mean of last five accuracy values
            mean_acc = np.median(acc[-5:])
            # Store results into the database
            self.cursor.execute('INSERT INTO wandb_run_metrics VALUES (?, ?, ?)', 
                                (run_id, gflops, mean_acc))
            self.conn.commit()

        data = {
            "gflops": gflops,
            "acc": mean_acc
        }
        return self._make_trial_data(trial_index, self.name, data[self.name])


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