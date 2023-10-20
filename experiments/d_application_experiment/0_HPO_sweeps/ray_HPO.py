from ast import main
from copy import deepcopy
from difflib import restore
import hydra
from omegaconf import OmegaConf
import ray
from ray import train, tune
from ray.train import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.ax import AxSearch
#from ray.tune.suggest.bayesopt import BayesOptSearch

import os
import sys

sys.path.append(f"{os.getcwd()}")
from experiments.ray_HPO import run_HPO
#os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

def get_search_space(
        name: str, 
        optimize_for: str, 
        additionals: dict, 
        dropout: bool = False,
    ):
    parameters = {
        "training.scheduler.patience": tune.randint(5, 100),
        "training.scheduler.factor": tune.loguniform(0.01, 0.5),
        "training.optimizer.lr": tune.loguniform(5e-5, 1e-2),
        "training.optimizer.weight_decay": tune.loguniform(1e-7, 1e-3),
    }
    if dropout:
        parameters["model.dropout_rate"] = tune.uniform(0.5, 0.7)

    parameters["ray"] = optimize_for
    parameters["wandb.tags"] = [name]

    # additionals
    for key, value in additionals.items():
        parameters[key] = value

    return parameters

def main():
    with hydra.initialize(config_path=".", version_base="1.2"):
        cfg = hydra.compose(config_name="HPO_ISIC2019", overrides=None)

    global_additionals = OmegaConf.to_container(cfg.additional_params, resolve=True, throw_on_missing=True)
    hpo_s = OmegaConf.to_container(cfg.HPOs, resolve=True, throw_on_missing=True)
    
    optimize_for = cfg.metric.name
    optimize_mode = cfg.metric.goal

    for name_hpo, hpo in hpo_s.items():
        if hpo.get("done", False):
            continue
        print("HPO for:", name_hpo)
        hpo_additionals = hpo.get("additional_params", {})
        additional_params = {**deepcopy(global_additionals), **hpo_additionals}
                
        param_space = get_search_space(
            name=name_hpo,
            optimize_for=optimize_for, 
            additionals=additional_params,
            dropout=hpo.get("dropout_rate", False),
        )

        grace_period = hpo.get("grace_period", cfg.optimizer.grace_period)
        num_trials = hpo.get("num_trials", cfg.optimizer.num_trials)

        run_HPO(
            name=name_hpo,
            param_space=param_space, 
            optimize_for=optimize_for,
            optimize_mode=optimize_mode,
            grace_period=grace_period,
            num_trials=num_trials,
            restore=hpo.get("restore", False),
        )


if __name__ == "__main__":
    main()