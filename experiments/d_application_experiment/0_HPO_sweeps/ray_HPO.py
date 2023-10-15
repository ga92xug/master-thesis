from ast import main
from copy import deepcopy
import ray
from ray import train, tune
from ray.train import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

import os
import sys
sys.path.append(f"{os.getcwd()}")
#os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
from training.main import hydra_initialize_init

def get_search_space(optimize_for: str, dropout: bool = False):
    parameters = {
        "training.scheduler.patience": tune.randint(5, 100),
        "training.scheduler.factor": tune.loguniform(0.01, 0.5),
        "training.optimizer.lr": tune.loguniform(5e-5, 1e-2),
        "training.optimizer.weight_decay": tune.loguniform(1e-7, 1e-3),
    }
    if dropout:
        parameters["model.dropout_rate"] = tune.uniform(0.5, 0.7)

    # additionals
    parameters["wandb.project"] = "SL-ray"
    parameters["ray"] = optimize_for
    parameters["training"] = "DeepDRiD-training"


    return parameters

def run_HPO(param_space: dict, optimize_for: str):
    trainable_with_gpu = tune.with_resources(hydra_initialize_init, {"gpu": 1})
    tune_config = tune.TuneConfig(
            metric=optimize_for,
            mode="max" if "acc" in optimize_for else "min",
            num_samples=1,
    )

    tuner = tune.Tuner(
        trainable_with_gpu,
        param_space=param_space,
        tune_config=tune_config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)




def run_HPO(
        param_space: dict, 
        optimize_for: str,
        grace_period: int,
        num_trials: int,
    ):
    trainable_with_gpu = tune.with_resources(hydra_initialize_init, {"gpu": 1})
    mode = "max" if "acc" in optimize_for else "min"
    
    asha_scheduler = ASHAScheduler(
            metric=optimize_for,
            mode=mode,
            grace_period=5,  # set grace period
            #max_t=100        # set maximum time for a trial
        )

    bayesopt = BayesOptSearch(
        metric=optimize_for,
        mode=mode,
    )

    tune.run(
        trainable_with_gpu,
        config=param_space,
        name="some_name",  # Set experiment name
        scheduler=asha_scheduler,
        search_alg=bayesopt,
        num_samples=num_trials  
    )



def main():
    optimize_for = "valid.acc"
    param_space = get_search_space(optimize_for, dropout=False)
    run_HPO(param_space, optimize_for)


if __name__ == "__main__":
    main()