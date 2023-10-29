from copy import deepcopy
from typing import Dict
import hydra
from omegaconf import DictConfig, OmegaConf
from ray import tune
import os
import sys
sys.path.append(f"{os.getcwd()}")
from experiments.d_application_experiment.ray_HPO import run_HPO


def epochs_based_on_initial_and_reduction_factor(
        initial_epochs: int, 
        reduction_factor: int,
    ) -> int:
    """
    Calculates the number of training epochs for low data regime.
    initial_epochs + (initial_epochs * (1/reduction_factor) * 0.25)
    
    Args:
        initial_epochs (int): initial number of epochs
        reduction_factor (int): low data regime reduction factor for dataset

    Returns:
        int: number of training epochs for low data regime
    """
    if reduction_factor == 1:
        return initial_epochs

    epochs = initial_epochs + ((initial_epochs * (1/reduction_factor)) * 0.25)
    return int(epochs)


def get_search_space(
        name: str, 
        optimize_for: str, 
        epochs: int,
        hydra_overrides: Dict, 
        dropout: bool = False,
        debug: bool = False,
    ) -> Dict:

    search_space = {
        # The BO engine can decide to disabled the scheduler by setting patience == epochs

        "training.scheduler.patience": tune.randint(int(0.05 * epochs), epochs),
        "training.scheduler.factor": tune.loguniform(0.01, 0.5),
        "training.optimizer.lr": tune.loguniform(5e-5, 1e-2),
        "training.optimizer.weight_decay": tune.loguniform(1e-7, 1e-3),
    }
    if dropout:
        search_space["model.dropout_rate"] = tune.uniform(0.5, 0.7)

    # Not searched for only trainings settings
    #if not debug:
    hydra_overrides["training.epochs"] = epochs
    hydra_overrides["training.earlystop.patience"] = int(epochs * 0.2) # 20% of epochs
    hydra_overrides["ray"] = optimize_for
    hydra_overrides["wandb.tags"] = [name]

    return search_space, hydra_overrides


def get_HPO_name(model_name: str, cfg: DictConfig):
    """
    Set the name for wandb and ray results.
    """
    global_overrides = cfg.dataset.global_overrides
    name = f"{model_name}_{cfg.dataset.name}_"\
        f"r{global_overrides['training.dataset.resolution']}_"\
        f"low_data_{global_overrides['training.dataset.reduction_factor']}"

    print("\nHPO for:", name)
    return name


def iterate_HPOs(cfg: DictConfig):
    global_overrides = OmegaConf.to_container(cfg.dataset.global_overrides, resolve=True, throw_on_missing=True)
    hpo_s = OmegaConf.to_container(cfg.HPOs, resolve=True, throw_on_missing=True)
    
    # optimization mode
    optimize_for = cfg.dataset.BO_optimizer.metric
    optimize_mode = cfg.dataset.BO_optimizer.goal


    epochs = epochs_based_on_initial_and_reduction_factor(
        initial_epochs=cfg.dataset.initial_epochs, 
        reduction_factor=cfg.dataset.global_overrides["training.dataset.reduction_factor"]
    )

    for model_name, hpo in hpo_s.items():
        # skip if done
        # hpo.get("done", False) and
        if hpo.get("eval_only", False):
            pass
        elif hpo.get("done", False):
            continue
        
        # name
        hpo_name = get_HPO_name(model_name, cfg)

        # hydra overrides
        model_overrides = hpo.get("model_overrides", {})
        hydra_overrides = {**model_overrides, **deepcopy(global_overrides)}
        
        # search space
        search_space, additional_overrides = get_search_space(
            name=hpo_name,
            optimize_for=optimize_for, 
            epochs=epochs,
            hydra_overrides=hydra_overrides,
            dropout=hpo.get("dropout_rate", False),
            debug=cfg.debug,
        )

        # optimization HPs
        grace_period = hpo.get("grace_period", cfg.dataset.BO_optimizer.grace_period)
        if isinstance(grace_period, float):
            grace_period = int(epochs * grace_period)
        num_trials = hpo.get("num_trials", cfg.dataset.BO_optimizer.num_trials)

        run_HPO(
            name=hpo_name,
            search_space=search_space,
            additional_overrides=additional_overrides, 
            optimize_for=optimize_for,
            optimize_mode=optimize_mode,
            grace_period=grace_period,
            num_trials=num_trials,
            epochs=epochs,
            restore=hpo.get("restore", False),
            debug=cfg.debug,
            only_eval=hpo.get("eval_only", False),
        )


@hydra.main(config_path=".", config_name="HPO", version_base="1.2")
def hydra_main(cfg: DictConfig) -> None:
    print(cfg)
    if cfg.debug:
        print(f"Debug level {cfg.debug}")
        global_overrides = cfg.dataset.global_overrides 
        global_overrides["wandb.mode"] = "disabled"
        cfg.dataset.initial_epochs = 2  
        cfg.dataset.BO_optimizer.num_trials = 2 
    iterate_HPOs(cfg)


if __name__ == "__main__":
    hydra_main()