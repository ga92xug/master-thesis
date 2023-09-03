import os
import shutil
import random
import timeit
import hydra
import torch
import wandb
#os.environ["WANDB_SILENT"] = "true"

import json
from omegaconf import DictConfig, OmegaConf

# Ax service
from ax.service.ax_client import AxClient, ObjectiveProperties

import logging
from ax.utils.common.logger import ROOT_STREAM_HANDLER
ROOT_STREAM_HANDLER.setLevel(logging.ERROR)

#from ax import save, load
from ax.core import Experiment, Data
# Save and load json
# from ax.storage.json_store.save import save_experiment
# from ax.storage.json_store.load import load_experiment
# from ax.storage.json_store.decoder import generation_strategy_from_json
# from ax.storage.json_store.encoder import object_to_json
#from ax.storage.json_store.encoder import generation_strategy_to_json
# Storage with sqa
from ax.storage.sqa_store.structs import DBSettings
from ax.storage.sqa_store.db import (
    get_engine, create_all_tables, 
    init_engine_and_session_factory,
)
from ax.storage.sqa_store.delete import delete_experiment
from ax.storage.sqa_store.save import save_experiment, save_generation_strategy
from ax.storage.registry_bundle import RegistryBundle
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.load import (
    load_experiment, 
    load_generation_strategy_by_experiment_name,
)

# Objective
from ax.core import MultiObjective, Objective, ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
# Generation Strategy
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
# Visualize
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.contour import plot_contour_plotly
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.service.utils.report_utils import exp_to_df
from ax.plot.contour import interact_contour_plotly
from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate
# check difference between the 2
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.plot.pareto_frontier import scatter_plot_with_pareto_frontier_plotly
from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly
# Models
from ax.modelbridge.registry import Models
from evaluate import evaluate

# Local
from runner_service import HydraWandbRunner
from search_space_service import Search_Space
from fetch_trial_data import TrialDataFetcher
from util import init_wandb


def warm_start(old_client_name: str, new_client: AxClient, max_building_time: int = None, initial: bool = False):
    """
    add the data from the previous client
    """
    if max_building_time is None:
        global cfg
        max_building_time = cfg.objective.max_building_time

    if os.path.exists(f"NAS/data/{old_client_name}/ax_client.json"):
        # backward compatibility
        old_client_file_path = f"NAS/data/{old_client_name}/ax_client.json"
    else:
        largest_version = get_largest_saved_version_ax_client()
        old_client_file_path = f"NAS/data/{old_client_name}/ax_client_{largest_version}.json"

    old_ax_client = AxClient.load_from_json_file(filepath=old_client_file_path)
    old_client_counts = get_count_trials(ax_client=old_ax_client)
    print("Old client counts: ", old_client_counts)

    df = exp_to_df(old_ax_client.experiment).sort_values(by=["trial_index"])
    # remove sobol trials only for this experiment
    remove_sobol = df[df["generation_method"] == "Sobol"].drop_duplicates(subset=["arm_name"])["arm_name"].values

    data_individual = old_ax_client.experiment.fetch_data().df
    counter = 0
    used_arm_names = set()
    for idx, (index, trial) in enumerate(old_ax_client.experiment.trials.items()):
        paramerization = trial.arm.parameters
        if trial.arm.name in used_arm_names:
            print("Trial already added: ", trial.arm.name)
            continue
        elif trial.arm.name in remove_sobol and initial:
            print("Trial from sobol: ", trial.arm.name)
            #pass
            continue
            
        data = {row["metric_name"]: (row["mean"], row["sem"]) for _index, row in data_individual[data_individual["trial_index"] == index].iterrows()}               
        if len(data) > 0:
            try:
                _parameterization, new_index = new_client.attach_trial(parameters=paramerization)
                add_data(ax_client=new_client, data=data, trial_index=new_index, step=idx, max_building_time=max_building_time, verbose=0, data_fetcher=None)
                used_arm_names.add(trial.arm.name)
            except:
                print("Could not attach trial: ", trial.arm.name)

            #self.log(raw_data, 0, idx)
        else:
            counter += 1
    count_trials = len(new_client.experiment.trials) + 1
    print("Warm start\n  number of trials without data:", counter, "\n  added:", len(new_client.experiment.trials))
    return count_trials


def add_data(
        ax_client: AxClient, 
        data: dict, 
        trial_index: int, 
        step: int, 
        max_building_time: int,
        data_fetcher: TrialDataFetcher,
        current_version: int = 0,
        verbose: int = 1,
    ):
    ax_data_keys = ["gflops", "valid_acc_weighted", "model_building_time"]
    ax_data = {k: data[k] for k in ax_data_keys}

    new_client = None
    if verbose >= 1:
        print("Add data: ", data)

    if len(ax_data) == 0:
        # abandon trial
        ax_client.abandon_trial(
            trial_index=trial_index, 
        )
        wandb.log({"Abandon trial": 1}, step=step)
        print("Abandon trial this behavior is not expected.")
        quit()
    elif len(data) in [1, 2]:
        if ax_data["model_building_time"] < max_building_time:
            wandb.log({"Bad trial": 1}, step=step)
            #data["GFLOPs"] = self.cfg.objective.bounds.gflops
            ax_data["model_building_time"] = max_building_time
            data["model_building_time"] = max_building_time
        # early stop trial 
        # expected if the model building time exceeds the limit

        ax_client.update_running_trial_with_intermediate_data(
            trial_index=trial_index,
            raw_data=ax_data,
        )
        ax_client.stop_trial_early(
            trial_index=trial_index,
        )
    else:
        # complete trial
        ax_client.complete_trial(
            trial_index=trial_index, 
            raw_data=ax_data,
        )

    log(data, step)
    
    if exp_to_df(ax_client.experiment)["arm_name"].duplicated().any() and data_fetcher is not None:
        # repeated trails bug https://github.com/facebook/Ax/issues/1704
        print("\nRepeated trials\n")
        ax_client, count_trials, num_trials = restart_ax_client(current_version=current_version)
        current_version += 1
        data_fetcher.connect_to_db(reset=True)
        return ax_client, current_version, count_trials, num_trials
    
    return new_client, current_version, None, None


def get_count_trials(ax_client: AxClient, num_sobol_trials: int = None, verbose: int = 0):
    if num_sobol_trials is None:
        global cfg
        num_sobol_trials = cfg.generation.num_sobol_trials
    if len(exp_to_df(ax_client.experiment)) == 0:
        return {
            "all_trials": 0,
            "sobol": 0,
            "full_bayesian": 0,
            "duplicates": 0,
        }
    df = exp_to_df(ax_client.experiment).sort_values(by=["trial_index"])
    df = df[df['trial_status'] != 'ABANDONED']

    count_manual = df[df["generation_method"] == "Manual"].drop_duplicates(subset=["arm_name"]).shape[0]
    count_sobol = df[df["generation_method"] == "Sobol"].drop_duplicates(subset=["arm_name"]).shape[0]
    count_full_bayesian = df[df["generation_method"] == "FullyBayesianMOO"].drop_duplicates(subset=["arm_name"]).shape[0]
    count_duplicates =  df['arm_name'].duplicated().sum()

    if num_sobol_trials <= count_manual:
        count_sobol = count_sobol + num_sobol_trials
        count_manual_bayesian = count_manual - num_sobol_trials
        count_full_bayesian = count_full_bayesian + count_manual_bayesian
    else:
        count_sobol = count_sobol + count_manual
        print("This case probably does not happen since the optimization never fails for sobol trials")

    count_all_trials = count_sobol + count_full_bayesian
    counts = {
        "all_trials": count_all_trials,
        "sobol": count_sobol,
        "full_bayesian": count_full_bayesian,
        "duplicates": count_duplicates,
    }
    if verbose >= 1:
        print("Number of trials: ", count_all_trials)
        print("Number of sobol trials: ", count_sobol)
        print("Number of full bayesian trials: ", count_full_bayesian)
        print("Number of duplicates: ", count_duplicates)
    
    return counts

def restart_ax_client(
        current_version: int,
    ):
    global save_folder

    old_client_name = f"{save_folder}/ax_client_{current_version}.json"
    old_ax_client = AxClient.load_from_json_file(filepath=old_client_name)
    counts = get_count_trials(ax_client=old_ax_client)

    if counts["sobol"] >= cfg.generation.num_sobol_trials:
        num_sobol_trials = 0
    else:
        num_sobol_trials = cfg.generation.num_sobol_trials - counts["sobol"]

    # Generation strategy
    generation_strategy = init_generation_strategy(num_sobol_trials)
    # setup ax client
    ax_client = AxClient(
        generation_strategy=generation_strategy,
        random_seed=cfg.seed,
    )
    num_trials = cfg.generation.num_total_trials
    old_client_name = f"{save_folder}/ax_client_{current_version}.json"

    # Experiment
    init_experiment(ax_client=ax_client)        
    count_trials = warm_start(old_client_name=old_client_name, new_client=ax_client)
    return ax_client, count_trials, num_trials


def init_new_ax_client(
        data_fetcher: TrialDataFetcher, 
    ):
    global cfg
    global save_folder
    restart_folder = cfg.client.restart
    wandb_save_path = f"{save_folder}/wandb_run_id.json"

    if not restart_folder:
        if not os.path.exists(restart_folder):
            ValueError("The ax_client.json file does not exist. Please set restart to True.")

        ax_client = AxClient.load_from_json_file(filepath=restart_folder)

        # number of trials
        ax_client.experiment.fetch_data()
        df = exp_to_df(ax_client.experiment).sort_values(by=["trial_index"])
        print(df)
        count_trials = df[df['trial_status'] != 'ABANDONED'].shape[0]
        num_trials = cfg.generation.num_total_trials

        # Get run id from json store
        with open(wandb_save_path, 'r') as f:
            wandb_run_id = json.load(f)['wandb_run_id']
        
        # resume wandb run
        _ = init_wandb(wandb_run_id, cfg)
        # connect to db
        data_fetcher.connect_to_db(reset=False)
        return ax_client, count_trials, num_trials
    else:
        # Generation strategy
        generation_strategy = init_generation_strategy()
        # setup ax client
        ax_client = AxClient(
            generation_strategy=generation_strategy,
            random_seed=cfg.seed,
        )
        # wandb
        wandb_config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        run = init_wandb(run_id=None, cfg=cfg, wandb_config=wandb_config)
        # connect to db
        data_fetcher.connect_to_db(reset=True)
        # Save the run_id
        with open(wandb_save_path, 'w') as f:
            json.dump({'wandb_run_id': run.id}, f)

        num_trials = cfg.generation.num_total_trials
        count_trials = 0
        # Experiment
        init_experiment(ax_client=ax_client)        
        if cfg.client.warm_start:
            # we want to keep the same cfg settings the script should know how to warm start
            count_trials = warm_start(old_client_name=cfg.client.warm_start, new_client=ax_client, initial=True)
    
    _ = get_count_trials(ax_client=ax_client)
    return ax_client, count_trials, num_trials

def init_generation_strategy(
        passed_sobol_trials: int = None,
    ):
    ######################################################################
    # Choosing the Generation Strategy
    # taken from https://github.com/facebook/Ax/issues/1454
    # how to deal with large search spaces
    global cfg
    device = torch.device('cuda' if torch.cuda.is_available() \
                               else "cpu")
    steps = []
    num_sobol_trials = passed_sobol_trials if passed_sobol_trials is not None else cfg.generation.num_sobol_trials
    if num_sobol_trials > 0:
        steps.append(
            GenerationStep(
                model=Models.SOBOL,
                num_trials=num_sobol_trials
            )
        )
    steps.append(
        GenerationStep(
                model=Models.FULLYBAYESIANMOO,
                num_trials=cfg.generation.num_fullbayesian_trials,
                model_kwargs={
                    "torch_device": device,
                    "num_samples": cfg.generation.num_samples,
                    "warmup_steps": cfg.generation.warmup_steps,
                    "disable_progbar": cfg.generation.disable_progbar, # Set to False to print a progress bar from MCMC
                },
                max_parallelism=1,
            )
    )
    generation_strategy=GenerationStrategy(
        name="SAASBO",
        steps=steps,
    )
    return generation_strategy

def init_runner():
    global cfg
    global save_folder
    # we have to convert the config to a dict because the config is not serializable
    # only matters for developer api
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    training_dict["NAS.max_gflops"] = cfg.objective.bounds.gflops
    training_dict["NAS.max_building_time"] = cfg.objective.max_building_time
    choice_2_range_params = OmegaConf.to_container(
        cfg.search_space.choice_2_range_params, resolve=True)
    
    hydra_wandb_runner = HydraWandbRunner(
        script_path=cfg.runner.script_path,
        wandb_entity=cfg.wandb.entity, 
        wandb_project=cfg.wandb.project,
        wandb_mode=cfg.wandb.mode_runs,
        db_path=save_folder,
        choice_2_range_param=choice_2_range_params,
        #strides=list(self.cfg.search_space.strides),
        training_dict=training_dict,
        verbose=cfg.runner.verbose,
    )
    return hydra_wandb_runner

def init_experiment(ax_client: AxClient):
    global cfg
    # search space
    eq_search_space = Search_Space(search_space_cfg=cfg.search_space)
    parameter = eq_search_space.get_parameters()
    parameter_constraints = eq_search_space.get_parameter_constraints()

    # new experiment
    ax_client.create_experiment(
        name=cfg.exp_name, 
        parameters=parameter,
        support_intermediate_data=True,
        objectives={
            # `threshold` arguments are optional
            "valid_acc_weighted": ObjectiveProperties(
                minimize=False, 
                threshold=cfg.objective.bounds.valid_acc
            ), 
            "gflops": ObjectiveProperties(
                minimize=True, 
                threshold=cfg.objective.bounds.gflops
            )
        },
        parameter_constraints=parameter_constraints,
        outcome_constraints=[f"model_building_time <= {cfg.objective.max_building_time - 1}"],
        tracking_metric_names=["model_building_time"],
        overwrite_existing_experiment=True,
        #is_test=True,
    )

def log(
        data: dict, 
        step: int, 
        verbose: int = 0
    ):
    for key, value in data.items():
        if len(value) > 1:
            data[key] = value[0]

    wandb.log(data, step=step, commit=True)
    generation_time = data.get("generation_time", None)

    if verbose >= 2 and generation_time is not None:
        print(f"Generation time: {generation_time:.2f} seconds")

def get_next_trial(ax_client: AxClient):
    # get next trial
    start = timeit.default_timer()
    trial = ax_client.get_next_trial()
    stop = timeit.default_timer()
    generation_time = stop - start
    return trial, generation_time

def get_largest_saved_version_ax_client(folder: str = None):
    if folder is None:
        global save_folder
        folder = save_folder
    # get the largest version
    version = 0
    for file in os.listdir(folder):
        if file.startswith("ax_client_"):
            file_version = int(file.split("_")[-1].split(".")[0])
            if file_version > version:
                version = file_version
    print("Largest version: ", version)
    return version

def add_fake_duplicate_trials(ax_client: AxClient, index: int):
    global cfg
    if not cfg.other.fake_duplicate_trials:
        return
    len_before = len(ax_client.experiment.trials)
    # add repeated trials
    data_df = ax_client.experiment.fetch_data().df
    fake_trial = ax_client.experiment.get_trials_by_indices([index])[0]

    # tuple has no attribute arm
    fake_paramerization = fake_trial.arm.parameters
    fake_raw_data = {row["metric_name"]: (row["mean"], row["sem"]) for _index, row in data_df[data_df["trial_index"] == index].iterrows()}
    _parameterization, new_index = ax_client.attach_trial(parameters=fake_paramerization, arm_name=fake_trial.arm.name)
    ax_client.complete_trial(trial_index=new_index,raw_data=fake_raw_data)
    print("\nTest duplicate trials before:", len_before, " after:", len(ax_client.experiment.trials), "\n")

def main_optim_loop(
        count_trials: int, 
        num_trials: int, 
        ax_client: AxClient,
        hydra_wandb_runner: HydraWandbRunner,
        data_fetcher: TrialDataFetcher,
    ):
    global cfg
    global save_folder
    verbose = cfg.other.verbose
    current_version = get_largest_saved_version_ax_client()
    ax_client_save_path = f"{save_folder}/ax_client_{current_version}.json"

    print("count_trials: ", count_trials, "/", num_trials)

    # Running optimization trials
    while count_trials < num_trials:
        if verbose >= 1:
            print(f"Trial: {count_trials}")

        # get next trial
        trial, generation_time = get_next_trial(ax_client=ax_client)

        # run trial
        trial_meta_data = hydra_wandb_runner.run(trial)
        trial_index=trial_meta_data["trial_index"]

        # fetch data
        data = data_fetcher.fetch_trial_data(
            trial_index=trial_index)
        data["generation_time"] = generation_time

        # sync data to Ax
        potential_new_client, new_version, potential_count_trials, potiential_num_trials = add_data(ax_client=ax_client, data=data, trial_index=trial_meta_data["trial_index"], step=count_trials, max_building_time=cfg.objective.max_building_time, data_fetcher=data_fetcher, current_version=current_version, verbose=verbose)
        if new_version != current_version:
            current_version = new_version
            ax_client_save_path = f"{save_folder}/ax_client_{current_version}.json"
            ax_client = potential_new_client
            count_trials = potential_count_trials
            num_trials = potiential_num_trials
            # Save
            ax_client.save_to_json_file(filepath=ax_client_save_path)

        # Save
        if count_trials % cfg.other.save_every == 0:
            ax_client.save_to_json_file(filepath=ax_client_save_path)

        # Test
        if cfg.other.fake_duplicate_trials:
            add_fake_duplicate_trials(ax_client=ax_client, index=0)
            
        if current_version > 1:
            # check if restarting was successful
            # number of trials improved
            print("Check if restarting was successful")
            previous_2_client_name = f"{save_folder}/ax_client_{current_version - 2}.json"
            previous_2_ax_client = AxClient.load_from_json_file(filepath=previous_2_client_name)
            previous_2_counts = get_count_trials(ax_client=previous_2_ax_client)
            current_counts = get_count_trials(ax_client=ax_client)

            if current_counts["all_trials"] <= previous_2_counts["all_trials"]:
                print("Restarting was not successful")
                quit()
            else:
                print("Restarting was successful")
                print("Previous 2 counts: ", previous_2_counts)
                print("Current counts: ", current_counts)
        
        count_trials += 1

    # final evaluation
    evaluate(ax_client=ax_client, step=count_trials)

def start_NAS():
    global cfg
    global save_folder
    save_folder = f"NAS/data/{cfg.exp_name}"
    if cfg.client.restart and os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    # Data fetcher
    data_fetcher = TrialDataFetcher(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        wandb_mode=cfg.wandb.mode,
        exp_name=cfg.exp_name,
        max_gflops=cfg.objective.bounds.gflops,
        max_building_time=cfg.objective.max_building_time,
        db_location=save_folder,
    )

    # Hydra wandb runner
    hydra_wandb_runner = init_runner()
    
    # init ax client
    ax_client, count_trials, num_trials = init_new_ax_client(
        data_fetcher=data_fetcher, 
    )

    # start optimization loop
    main_optim_loop(
        count_trials=count_trials, 
        num_trials=num_trials, 
        ax_client=ax_client,
        hydra_wandb_runner=hydra_wandb_runner,
        data_fetcher=data_fetcher,
    )    


@hydra.main(config_path="conf", config_name="nas", version_base="1.2")
def main(config: DictConfig) -> None:
    global cfg
    cfg = config

    if cfg.other.debug:
        print("Debug mode")
        cfg.generation.num_samples = 16 #128
        cfg.generation.warmup_steps = 32 # 256
        cfg.generation.disable_progbar = False
        # Temporarily disable struct mode to add params
        #OmegaConf.set_struct(cfg.training, False)
        #cfg.training.training.epochs = 1
        #cfg.training.training.steps_per_epoch = 10
        #OmegaConf.set_struct(cfg.training, True)
        cfg.wandb.mode = "disabled"

    #nas = NAS(cfg)
    if cfg.evaluate_only:
        pass
        #evaluate(ax_client=nas.ax_client, step=200)
    else:
        start_NAS()


if __name__ == "__main__":
    main()