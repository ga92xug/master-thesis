import os
from typing import Dict, List
import yaml
import pandas as pd

def get_from_logged_hps(base_path: str, hyperparameter_names: List[str]) -> Dict[str, float]:
    """
    Load hyperparameters from a YAML file.

    Parameters:
    - hyperparameters_file: Path to the YAML file containing hyperparameters.

    Returns:
    - dict: A dictionary of hyperparameters.
    """
    path = os.path.join(base_path, "hparams.yaml")
    with open(path, 'r') as file:
        hyperparameters = yaml.safe_load(file)

    fetched_hyperparameters = {hp: hyperparameters.get(hp) for hp in hyperparameter_names}

    return fetched_hyperparameters

def get_metrics(base_path: str, metric_name: str) -> Dict[str, float]:
    """
    Load metrics from a CSV file.

    Parameters:
    - metrics_file: Path to the CSV file containing metrics.

    Returns:
    - pandas.DataFrame: A DataFrame of metrics.
    """
    path = os.path.join(base_path, "metrics.csv")
    metrics = pd.read_csv(path)
    if metric_name in metrics.columns:
        # get last non-NaN value
        metric_value = metrics[metric_name].dropna().iloc[-1]
    else:
        metric_value = {metric_name: None}

    return metric_value


def fetch_trial_data(
        trial_index: int,
        base_path: int, 
        metric_name: str, 
        hyperparameter_names: List[str] = ['GFLOPs_per_image', 'net_building_time'],
    ):
    """
    Fetch trial data including a specified metric and hyperparameters.

    Parameters:
    - hyperparameters_file: Path to the YAML file with hyperparameters.
    - metrics_file: Path to the CSV file with metrics.
    - metric_name: The name of the metric to fetch.
    - hyperparameter_names: A list of hyperparameter names to fetch.

    Returns:
    - dict: A dictionary with the requested metric and hyperparameters.
    """
    base_path = os.path.join(base_path, "csv", f"version_{trial_index}")

    hp_logged_metrics = get_from_logged_hps(base_path, hyperparameter_names)
    metrics = get_metrics(base_path, metric_name)

    combined = {**hp_logged_metrics, metric_name: metrics}
    return combined

