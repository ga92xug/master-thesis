from typing import Dict, Optional
import wandb
import numpy as np
import sqlite3
import time

class RunDataFetcher():
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
            db_location: str = "NAS/data/"
        ):
        """Initializes the RunDataFetcher with the entity, project, wandb_mode, experiment name, and database location."""
        self.project = project
        self.entity = entity
        self.db_location = db_location
        self.exp_name = exp_name
        self.db_file = db_location + exp_name + "/wandb_cache.db"
        self.wandb_mode = wandb_mode

    def connect_to_db(self):
        """Establishes a connection to the database and creates a table if it does not exist."""
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
    
    def fetch_run_data(self, wandb_run_id: str) -> Dict:
        """
        Fetch data from Weights & Biases (wandb) run or from local database.

        :param wandb_run_id: wandb run id to fetch data from.
        :returns: A dictionary with 'gflops', 'val_acc', 'train_duration', 'val_duration' keys.
        """

        print(f"Fetching data for run {wandb_run_id}")

        # Attempt to fetch data from local DB
        run_data = self._fetch_from_db(wandb_run_id)
        if run_data is None and self.wandb_mode != "disabled":
            # If data not in DB and wandb is enabled, fetch data from wandb
            run_data = self._fetch_from_wandb(wandb_run_id)
        elif run_data is None:
            # If data not in DB and wandb is disabled, generate random data
            print("Wandb is disabled, returning random values")
            run_data = {key: val for key, val in zip(['gflops', 'val_acc', 'train_duration', 'val_duration'], np.random.rand(4))}

        return run_data

    def _fetch_from_db(self, wandb_run_id: str) -> Optional[Dict]:
        """
        Fetch data from the local database.

        :param wandb_run_id: wandb run id to fetch data from.
        :returns: A dictionary with 'gflops', 'val_acc', 'train_duration', 'val_duration' keys or None if data doesn't exist.
        """
        conn, cursor = self.connect_to_db()
        cursor.execute('SELECT * FROM wandb_run_metrics WHERE run_id = ?', (wandb_run_id,))
        result = cursor.fetchone()

        if result is not None:
            return {key: float(val) for key, val in zip(['gflops', 'val_acc', 'train_duration', 'val_duration'], result[1:])}
        else:
            return None

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
        cursor.execute('INSERT INTO wandb_run_metrics VALUES (?, ?, ?, ?, ?)', 
                    (wandb_run_id, metrics['gflops'], metrics['val_acc'], metrics['train_duration'], metrics['val_duration']))
        conn.commit()