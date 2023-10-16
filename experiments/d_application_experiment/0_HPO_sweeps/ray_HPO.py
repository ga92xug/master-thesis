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
#os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
from training.main import hydra_initialize_init

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

def run_HPO_old(param_space: dict, optimize_for: str):
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
        name: str,
        param_space: dict, 
        optimize_for: str,
        optimize_mode: str,
        grace_period: int,
        num_trials: int,
        restore: bool = False,
    ):
    print("param_space", param_space)

    trainable_with_gpu = tune.with_resources(hydra_initialize_init, {"gpu": 1})

    algo = AxSearch(
        #parameter_constraints=["x1 + x2 <= 2.0"],
        #outcome_constraints=["l2norm <= 1.25"],
    )

    asha_scheduler = ASHAScheduler(
            grace_period=grace_period,  
        )

    tune_config=tune.TuneConfig(
        metric=optimize_for,
        mode=optimize_mode,
        search_alg=algo,
        scheduler=asha_scheduler,
        num_samples=num_trials,
    )

    run_config=train.RunConfig(
        name=name,
    )
    
    if restore:
        tuner = tune.Tuner.restore(
            os.path.expanduser(f"~/ray_results/{name}"),
            trainable=trainable_with_gpu,
            resume_unfinished=True,
            resume_errored=True,
        )
    else:
        if os.path.exists(os.path.expanduser(f"~/ray_results/{name}")):
            # remove old results
            os.system(f"rm -rf ~/ray_results/{name}")
        tuner = tune.Tuner(
            trainable_with_gpu,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
        
    #print("resume")
    #tune.Tuner.restore()
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    #bayesopt = BayesOptSearch(
    #    metric=optimize_for,
    #    mode=optimize_mode,
    #)



def main():
    with hydra.initialize(config_path=".", version_base="1.2"):
        cfg = hydra.compose(config_name="HPO", overrides=None)

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
        #print("param_space", param_space)
        run_HPO(
            name=name_hpo,
            param_space=param_space, 
            optimize_for=optimize_for,
            optimize_mode=optimize_mode,
            grace_period=hpo["grace_period"],
            num_trials=hpo["num_trials"],
            restore=hpo.get("restore", False),
        )


if __name__ == "__main__":
    main()