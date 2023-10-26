from typing import Dict
import hydra
import numpy as np
import ray
from ray import train, tune
from ray.tune.stopper.stopper import Stopper
from ray.tune.stopper import ExperimentPlateauStopper
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.ax import AxSearch
from ray.tune.search import ConcurrencyLimiter

import os
import sys

sys.path.append(f"{os.getcwd()}")
#os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
from training.main import hydra_initialize_init
from experiments.d_application_experiment.utils_HPO import HPOEarlyStopper

def run_HPO(
        name: str,
        search_space: dict, 
        additional_overrides: dict,
        optimize_for: str,
        optimize_mode: str,
        grace_period: int,
        num_trials: int,
        epochs: int,
        restore: bool = False,
        experiment_level_early_stop: bool = False,
        debug: bool = False,
        only_eval: bool = False,
    ):
    print("Search space:", search_space)
    print("Additional overrides:", additional_overrides)

    trainable_with_resources = tune.with_resources(hydra_initialize_init, {"gpu": 1, "cpu": 28})
    trainable_with_parameters = tune.with_parameters(trainable_with_resources, 
        additional_overrides=additional_overrides)

    algo = AxSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=2)
    asha_scheduler = ASHAScheduler(time_attr="training_iteration", 
                                   max_t=epochs, grace_period=grace_period)

    tune_config=tune.TuneConfig(
        metric=optimize_for,
        mode=optimize_mode,
        search_alg=algo,
        scheduler=asha_scheduler,
        num_samples=num_trials,
    )

    stopper = None
    if experiment_level_early_stop:
        stopper = HPOEarlyStopper(
            metric=optimize_for, 
            mode=optimize_mode, 
            patience=5, 
            min_delta=0.001,
            min_num_trials=max(num_trials//2, 10)
        )

    run_config=train.RunConfig(
        name=name, 
        stop=stopper,
        log_to_file=True,
    )
    
    if restore or only_eval:
        tuner = tune.Tuner.restore(
            path=os.path.expanduser(f"~/ray_results/{name}"),
            trainable=trainable_with_parameters,
            resume_unfinished=True,
            restart_errored=True,
            #resume_errored=True,
        )
    else:
        # remove old results
        if os.path.exists(os.path.expanduser(f"~/ray_results/{name}")):
            os.system(f"rm -rf ~/ray_results/{name}")
        tuner = tune.Tuner(
            trainable_with_parameters,
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )
    
    if debug:
        print("Debugging...")
        return

    if only_eval:
        print("Only evaluating...")
        results = tuner.get_results()
    else:
        results = tuner.fit()

    best_config = results.get_best_result().config
    print("Best hyperparameters found were: ", results.get_best_result().config)
    run_best_HP_with_seeds(best_config, additional_overrides)

    
def run_best_HP_with_seeds(best_HPs: Dict, additional_overrides: Dict):
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    # remove ray from additional_overrides
    additional_overrides.pop("ray", None)
    additional_overrides["wandb.tags"] = ["low_data"]
    additional_overrides["wandb.project"] = "SL-Application"

    for seed in range(5):
        additional_overrides["other.seed"] = seed
        hydra_initialize_init(best_HPs, additional_overrides)