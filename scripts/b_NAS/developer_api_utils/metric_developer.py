import wandb
from ax.core.data import Data
import pandas as pd
from ax.core.metric import Metric
from ax.core.outcome_constraint import ComparisonOp
from ax.core.types import Tuple
from ax.utils.common.result import Err, Ok
import numpy as np
import sqlite3
import time

import sys
sys.path.append("../NAS")
from test_util.trial_dummy import Trial_Dummy

# currently we use an evaluation function instead by we might need the 
# flexibility of a metric later
class WandbMetric(Metric):
    """
    A metric that fetches data from a wandb run.
    Implements a database cache to avoid repeated API calls.
    """

    def __init__(self, name: str, entity: str, project: str, wandb_mode: str,
                lower_is_better: bool, exp_name: str, 
                db_location: str = "NAS/data/"):
        super().__init__(name, lower_is_better=lower_is_better)
        self.project = project
        self.entity = entity
        self.db_location = db_location
        self.exp_name = exp_name
        self.db_file = db_location + exp_name + "/wandb_cache.db"
        self.wandb_mode = wandb_mode

    def connect_to_db(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wandb_run_metrics (
                run_id TEXT PRIMARY KEY,
                gflops REAL,
                acc_last_five TEXT,
                train_duration REAL,
                valid_duration REAL
            )
        ''')
        conn.commit()
        return conn, cursor
    

    def fetch_trial_data(self, trial_index, wandb_run_id) -> Tuple:
        api = wandb.Api()
        conn, cursor = self.connect_to_db()

        cursor.execute('SELECT * FROM wandb_run_metrics WHERE run_id = ?', (trial_index,))
        result = cursor.fetchone()

        if result is not None:
            gflops, mean_acc, train_duration, valid_duration = float(result[1]), float(result[2]), float(result[3]), float(result[4])
        elif self.wandb_mode == "disabled":
            # if wandb is disabled, we just return random values for testing
            print("Wandb is disabled, returning random values")
            gflops, mean_acc, train_duration, valid_duration = np.random.rand(4)
        else:
            successful_fetch = False
            while not successful_fetch:
                try:
                    print("Fetching data from wandb: ", f"{self.entity}/{self.project}/runs/{wandb_run_id}")
                    run = api.run(f"{self.entity}/{self.project}/runs/{wandb_run_id}")
                    gflops = run.history(keys=['GFLOPs']).values[:,1][0]
                    acc = run.history(keys=['valid.acc']).values[:,1]
                    train_durations = run.history(keys=['train.duration']).values[:,1]
                    valid_durations = run.history(keys=['valid.duration']).values[:,1]
                    train_duration = np.median(train_durations)
                    valid_duration = np.median(valid_durations)
                    mean_acc = np.median(acc[-5:])
                    cursor.execute('INSERT INTO wandb_run_metrics VALUES (?, ?, ?, ?, ?)', 
                                   (trial_index, gflops, mean_acc, train_duration, valid_duration))
                    conn.commit()
                    successful_fetch = True
                except:
                    print("Error fetching data from wandb, trying again in 5 seconds")
                    time.sleep(5) 

        data = {
            "gflops": gflops,
            "val_acc": mean_acc,
            "train_duration": train_duration,
            "val_duration": valid_duration
        }
        #print("fetched data", data)
        return data
        #return self._make_trial_data(trial_index, self.name, data[self.name])


    def to_json(self):
        return {
            "name": self.name,
            "lower_is_better": self.lower_is_better,
            "entity": self.entity,
            "project": self.project,
            "exp_name": self.exp_name,
            "db_location": self.db_location,
        }

    @classmethod
    def from_json(cls, json_repr):
        return cls(
            name=json_repr["name"],
            lower_is_better=json_repr["lower_is_better"],
            entity=json_repr["entity"],
            project=json_repr["project"],
            exp_name=json_repr["exp_name"],
            db_location=json_repr["db_location"],
        )



    def _make_trial_data(self, trial_index, metric_name, metric_value):
        """
        Convert fetched data into AX's TrialData format.
        """
        print("Making trial data")
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
        #return df
        return Ok(Data(df=df))

    
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
        result = None

        if self.wandb_mode == "disabled":
            # if wandb is disabled, we just return random values for testing
            print("Wandb is disabled, returning random values")
            return {
                "valid.acc": np.random.rand(),
                "gflops": np.random.rand(),
                "train_duration": np.random.rand(),
                "valid_duration": np.random.rand()
            }
        
        while result is None and not tried_fetching:
            cursor.execute('SELECT * FROM wandb_run_metrics WHERE run_id = ?', (trial.index,))
            result = cursor.fetchone()
            if result is not None:
                print("found trial data")
                gflops, mean_acc, train_duration, valid_duration = float(result[1]), float(result[2]), float(result[3]), float(result[4])
                metrics = {
                    "val_acc": mean_acc,
                    "gflops": gflops,
                    "train_duration": train_duration,
                    "valid_duration": valid_duration
                }
                return metrics
            else:
                self.fetch_trial_data(trial)
                tried_fetching = True

        raise ValueError("Trial data not found in db")


if __name__ == "__main__":
    fake_trial = Trial_Dummy(1)    

    # test
    metric = WandbMetric(name="valid.acc", entity="automl", project="nas", wandb_mode="disabled",
                         lower_is_better=False, exp_name="mnist_rot")
    result = metric.get_metrics(fake_trial)
    print(result)


        # Connect to the database
    conn = sqlite3.connect(metric.db_file)
    cursor = conn.cursor()

    # Execute the SELECT statement to fetch all rows from the table
    cursor.execute("SELECT * FROM wandb_run_metrics")

    # Fetch all rows from the result
    rows = cursor.fetchall()

    # Print the column names
    column_names = [description[0] for description in cursor.description]
    print(column_names)

    # Print the content of the table
    for row in rows:
        print(row)

    # Close the cursor and connection
    cursor.close()
    conn.close()
    metric.connect_to_db()