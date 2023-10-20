import ray
from ray import train, tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.ax import AxSearch

import os
import sys
sys.path.append(f"{os.getcwd()}")
#os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
from training.main import hydra_initialize_init


def run_HPO(
        name: str,
        search_space: dict, 
        additional_overrides: dict,
        optimize_for: str,
        optimize_mode: str,
        grace_period: int,
        num_trials: int,
        restore: bool = False,
        debug: bool = False,
    ):
    print("Search space:", search_space)
    print("Additional overrides:", additional_overrides)

    trainable_with_resources = tune.with_resources(hydra_initialize_init, {"gpu": 1})
    trainable_with_parameters = tune.with_parameters(trainable_with_resources, 
        additional_overrides=additional_overrides)

    algo = AxSearch()
    asha_scheduler = ASHAScheduler(grace_period=grace_period)

    tune_config=tune.TuneConfig(
        metric=optimize_for,
        mode=optimize_mode,
        search_alg=algo,
        scheduler=asha_scheduler,
        num_samples=num_trials,
    )

    run_config=train.RunConfig(name=name)
    
    if restore:
        tuner = tune.Tuner.restore(
            os.path.expanduser(f"~/ray_results/{name}"),
            trainable=trainable_with_parameters,
            resume_unfinished=True,
            resume_errored=True,
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

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
