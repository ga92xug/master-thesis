from typing import Dict, Optional
import wandb
import numpy as np
import sqlite3
import time

class TrialDataFetcher():
    """
    The RunDataFetcher class is used to retrieve run data from the Weights & Biases (wandb) tool.
    It incorporates a database cache to prevent excessive API calls.

    Attributes:
        entity (str): The name of the wandb entity.
        project (str): The name of the wandb project.
        wandb_mode (str): The mode of wandb, which can be "disabled" for testing purposes.
        exp_name (str): The name of the experiment.
        db_location (str): The location of the database. Default is "NAS/data/".
        db_file (str): The file path to the database file.
    """

    def __init__(
            self, 
            entity: str, 
            project: str, 
            wandb_mode: str,
            exp_name: str, 
            max_gflops: int,
            db_location: str = "NAS/data/"
        ):
        """Initializes the RunDataFetcher with the entity, project, wandb_mode, experiment name, and database location."""
        self.project = project
        self.entity = entity
        self.db_location = db_location
        self.exp_name = exp_name
        self.db_file = db_location + "/trial_cache.db"
        self.wandb_mode = wandb_mode
        self.max_gflops = max_gflops

    def connect_to_db(self, reset: bool = False):
        """Establishes a connection to the database, deletes the table if it exists and creates a new one."""
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()
        
        if reset:
            # Drop the table if it already exists
            self.cursor.execute('''
                DROP TABLE IF EXISTS run_metrics
            ''')
            self.conn.commit()

        # Create a new table
        self.cursor.execute('''
            CREATE TABLE run_metrics (
                trial_index TEXT PRIMARY KEY,
                gflops REAL,
                valid_acc REAL,
                train_duration REAL,
                valid_duration REAL
            )
        ''')
        self.conn.commit()

    
    def fetch_trial_data(self, trial_index: str) -> Dict:
        """
        Fetch data from Weights & Biases (wandb) run or from local database.

        :param wandb_run_id: wandb run id to fetch data from.
        :returns: A dictionary with 'gflops', 'val_acc', 'train_duration', 'val_duration' keys.
        """

        # Attempt to fetch data from local DB
        run_data = self._fetch_from_db(trial_index)
        """
        if run_data is None and self.wandb_mode != "disabled":
            # If data not in DB and wandb is enabled, fetch data from wandb
            run_data = self._fetch_from_wandb(trial_index)
        elif run_data is None:
            # If data not in DB and wandb is disabled, generate random data
            print("Wandb is disabled, returning random values")
            run_data = {key: val for key, val in zip(['gflops', 'val_acc', 'train_duration', 'val_duration'], np.random.rand(4))}
        """
        return run_data

    def _fetch_from_db(self, trial_index: str) -> Optional[Dict]:
        """
        Fetch data from the local database.

        :param wandb_run_id: wandb run id to fetch data from.
        :returns: A dictionary with 'gflops', 'val_acc', 'train_duration', 'val_duration' keys or None if data doesn't exist.
        """
        self.cursor.execute('SELECT * FROM run_metrics WHERE trial_index = ?', (trial_index,))
        result = self.cursor.fetchone()

        if result is not None:
            result_dict = {}
            for key, val in zip(['gflops', 'valid_acc', 'train_duration', 'valid_duration'], result[1:]):
                if key == 'gflops' and val is None:
                    ValueError(f"Trial {trial_index} does not have GFLOPs data. We always expect an estimate of GFLOPs. Even if trial failed.")
                try:
                    result_dict[key] = float(val)
                except:
                    if result_dict['gflops'] <= self.max_gflops:
                        raise ValueError(f"Trial {trial_index} has invalid {key} value: {val}. We only accept missing values if GFLOPs is above {self.max_gflops}.")
                    result_dict[key] = None
                    
            return result_dict
        else:
            ValueError(f"Trial {trial_index} not found in database")

    def _fetch_from_wandb(self, wandb_run_id: str) -> Dict:
        """
        Fetch data from the Weights & Biases (wandb) API.

        :param wandb_run_id: wandb run id to fetch data from.
        :returns: A dictionary with 'gflops', 'val_acc', 'train_duration', 'val_duration' keys.
        """
        api = wandb.Api()
        successful_fetch = False
        while not successful_fetch:
            try:
                print(f"Fetching data from wandb: {self.entity}/{self.project}/runs/{wandb_run_id}")
                run = api.run(f"{self.entity}/{self.project}/runs/{wandb_run_id}")
                metrics = self._get_wandb_metrics(run)
                self._store_to_db(wandb_run_id, metrics)
                successful_fetch = True
            except:
                print("Error fetching data from wandb, trying again in 5 seconds")
                time.sleep(5) 
        return metrics

    def _get_wandb_metrics(self, run) -> Dict:
        """
        Extract relevant metrics from a wandb run object.

        :param run: A wandb run object.
        :returns: A dictionary with 'gflops', 'val_acc', 'train_duration', 'val_duration' keys.
        """
        print("run.history(keys=['GFLOPs'])", run.history(keys=['GFLOPs']))
        gflops = run.history(keys=['GFLOPs']).values[:,1][0]
        acc = run.history(keys=['valid.acc']).values[:,1]
        train_durations = run.history(keys=['train.duration']).values[:,1]
        valid_durations = run.history(keys=['valid.duration']).values[:,1]
        train_duration = np.median(train_durations)
        valid_duration = np.median(valid_durations)
        mean_acc = np.median(acc[-5:])
        return {
            'gflops': gflops, 
            'val_acc': mean_acc, 
            'train_duration': train_duration, 
            'val_duration': valid_duration
        }

    def _store_to_db(self, wandb_run_id: str, metrics: Dict) -> None:
        """
        Store run data into local database.

        :param wandb_run_id: wandb run id of the data.
        :param metrics: A dictionary with 'gflops', 'val_acc', 'train_duration', 'val_duration' keys.
        """
        conn, cursor = self.connect_to_db()
        cursor.execute('INSERT INTO run_metrics VALUES (?, ?, ?, ?, ?)', 
                    (wandb_run_id, metrics['gflops'], metrics['val_acc'], metrics['train_duration'], metrics['val_duration']))
        conn.commit()