from typing import Dict
import wandb
import yaml
import os

from ray.tune import Tuner



import os
import sys

sys.path.append(f"{os.getcwd()}")
#os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
from training.main import hydra_initialize_init

def stats_from_experiment(path_to_experiment: str) -> Dict:
    experiment_count = 0
    good_experiment_count = 0
    for run in os.listdir(path_to_experiment):
        if "hydra_initialize_init_" not in run:
            continue

        experiment_count += 1

        run_path = os.path.join(path_to_experiment, run)
        progress_path = os.path.join(run_path, "progress.csv")
        result_path = os.path.join(run_path, "result.json")

        if not os.path.exists(progress_path) or not os.path.exists(result_path):
            print("Missing progress or result file", run)
            continue

        good_experiment_count += 1

    stats = {
        "experiment_count": experiment_count,
        "success": good_experiment_count,
        "failure": experiment_count - good_experiment_count,
    }
    print(stats)
    return stats


def main():
    path_to_ray = os.path.expanduser(f"~/ray_results/")

    for experiment in os.listdir(path_to_ray):
        if not ("128" in experiment and "low_data" in experiment):
            continue
        print(experiment)

        path_to_experiment = os.path.join(path_to_ray, experiment)

        stats = stats_from_experiment(path_to_experiment)

        if stats["experiment_count"] < 20:
            print("Not enough experiments")
            continue

        
        tuner = Tuner.restore(path_to_experiment, trainable=hydra_initialize_init)
        results = tuner.get_results()
        best_config = results.get_best_result().config
        print(best_config)

    #tuner = Tuner.restore('/ray_results/to/experiment', trainable=hydra_initialize_init)
    #results = tuner.get_results()
    #print(results)


if __name__ == "__main__":
    main()
