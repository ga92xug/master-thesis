from itertools import count
import os
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


def warm_start(old_client_name: str, new_client: AxClient, initial: bool = False):
    # we add the data from the previous client
    old_client_file_path = f"NAS/data/{old_client_name}/ax_client.json"
    old_ax_client = AxClient.load_from_json_file(filepath=old_client_file_path)

    df = exp_to_df(old_ax_client.experiment).sort_values(by=["trial_index"])
    # remove sobol trials only for this experiment
    remove_sobol = df[df["generation_method"] == "Sobol"].drop_duplicates(subset=["arm_name"])["arm_name"].values

    data_individual = old_ax_client.experiment.fetch_data().df
    counter = 0
    used_arm_names = set()
    for idx, (index, trial) in enumerate(old_ax_client.experiment.trials.items()):
        #print("Trial: ", trial)
        paramerization = trial.arm.parameters
        if trial.arm.name in used_arm_names:
            print("Trial already added: ", trial.arm.name)
            continue
        elif trial.arm.name in remove_sobol and initial:
            print("Trial from sobol: ", trial.arm.name)
            continue
        else:
            used_arm_names.add(trial.arm.name)
        raw_data = {row["metric_name"]: (row["mean"], row["sem"]) for _index, row in data_individual[data_individual["trial_index"] == index].iterrows()}               
        if len(raw_data) > 0:
            try:
                _parameterization, new_index = new_client.attach_trial(parameters=paramerization)
                add_data(ax_client=new_client, data=raw_data, trial_index=new_index, step=idx, max_building_time=None)
            except:
                print("Could not attach trial: ", paramerization)
            #self.log(raw_data, 0, idx)
        else:
            counter += 1
    count_trials = len(new_client.experiment.trials) + 1
    print("Number of trials without data: ", counter)
    print("Added ", len(new_client.experiment.trials), "trials to the experiment")
    quit()

    return count_trials


def add_data(ax_client: AxClient, data: dict, trial_index: int, step: int, max_building_time: int):
    print("Add data: ", data)
    if len(data) == 0:
        # abandon trial
        ax_client.abandon_trial(
            trial_index=trial_index, 
        )
        wandb.log({"Abandon trial": 1}, step=step)
        print("Abandon trial this behavior is not expected.")
        quit()
    elif len(data) in [1, 2]:
        if data["model_building_time"] < max_building_time:
            wandb.log({"Bad trial": 1}, step=step)
            #data["GFLOPs"] = self.cfg.objective.bounds.gflops
            data["model_building_time"] = max_building_time
        # early stop trial 
        # expected if the model building time exceeds the limit
        ax_client.update_running_trial_with_intermediate_data(
            trial_index=trial_index,
            raw_data=data,
        )
        ax_client.stop_trial_early(
            trial_index=trial_index,
        )
    else:
        # complete trial
        ax_client.complete_trial(
            trial_index=trial_index, 
            raw_data=data,
        )
    
    if exp_to_df(ax_client.experiment).duplicated().any():
        # repeated trails bug https://github.com/facebook/Ax/issues/1704
        print("Repeated trials")
        init_new_ax_client(
            data_fetcher=None, 
            restart_folder=None, 
            save_folder=None, 
            current_version=None, 
            wandb_run_id=None, 
            initial=False
        )


def init_new_ax_client(
        #cfg: DictConfig, 
        data_fetcher: TrialDataFetcher, 
        restart_folder: str,
        save_folder: str,
        current_version: int, 
        wandb_run_id: str, 
        initial: bool = False,
    ):
    global cfg

    if not cfg.client.restart and initial:
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
        with open(wandb_run_id, 'r') as f:
            wandb_run_id = json.load(f)['wandb_run_id']
        
        # resume wandb run
        run = init_wandb(wandb_run_id, cfg)
        # connect to db
        data_fetcher.connect_to_db(reset=False)
        return ax_client, count_trials, num_trials
    
    # Generation strategy
    generation_strategy = init_generation_strategy()
    # setup ax client
    ax_client = AxClient(
        generation_strategy=generation_strategy,
        random_seed=cfg.seed,
    )

    if initial:
        wandb_config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        # init wandb
        run = init_wandb(run_id=None, cfg=cfg, wandb_config=wandb_config)
        # connect to db
        data_fetcher.connect_to_db(reset=True)
        # Save the run_id
        with open(wandb_run_id, 'w') as f:
            json.dump({'wandb_run_id': run.id}, f)

        num_trials = cfg.generation.num_total_trials

        # Experiment
        init_experiment(cfg=cfg, ax_client=ax_client)        
        if cfg.client.warm_start:
            count_trials = warm_start(old_client_name=cfg.client.warm_start, new_client=ax_client)

    else:
        num_trials = cfg.generation.num_total_trials
        old_client_name = f"{save_folder}/ax_client_{current_version}.json"

        # Experiment
        init_experiment(cfg=cfg, ax_client=ax_client)        
        count_trials = warm_start(old_client_name=old_client_name, new_client=ax_client)

    

    return ax_client, count_trials, num_trials



def init_generation_strategy():
    ######################################################################
    # Choosing the Generation Strategy
    # taken from https://github.com/facebook/Ax/issues/1454
    # how to deal with large search spaces
    global cfg
    device = torch.device('cuda' if torch.cuda.is_available() \
                               else "cpu")
    steps = []
    if cfg.generation.num_sobol_trials > 0:
        steps.append(
            GenerationStep(
                model=Models.SOBOL,
                num_trials=cfg.generation.num_sobol_trials
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
                    "disable_progbar": cfg.generation.progress_bar, # Set to False to print a progress bar from MCMC
                },
                max_parallelism=1,
            )
    )
    generation_strategy=GenerationStrategy(
        name="SAASBO",
        steps=steps,
    )
    return generation_strategy

def init_runner(save_folder: str):
    global cfg
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

def log(raw_data, generation_time, step, verbose: int = 0):
    raw_data["generation_time"] = generation_time
    wandb.log(raw_data, step=step, commit=True)

    if verbose >= 2:
        print(f"Generation time: {generation_time:.2f} seconds")

def get_next_trial(ax_client: AxClient):
    # get next trial
    start = timeit.default_timer()
    trial = ax_client.get_next_trial()
    stop = timeit.default_timer()
    generation_time = stop - start
    return trial, generation_time


def main_optim_loop(
        count_trials: int, 
        num_trials: int, 
        ax_client: AxClient,
        hydra_wandb_runner: HydraWandbRunner,
        data_fetcher: TrialDataFetcher,
        save_folder: str,
        current_version: int,
        ):
    global cfg
    verbose = cfg.other.verbose
    ax_client_save_path = f"{save_folder}/ax_client_{current_version}.json"

    # Running optimization trials
    while count_trials < num_trials:
        if verbose >= 1:
            print(f"Trial: {count_trials}")

        # get next trial
        trial, generation_time = get_next_trial()

        # run trial
        trial_meta_data = hydra_wandb_runner.run(trial)

        # fetch data
        ax_data, raw_data, = data_fetcher.fetch_trial_data(
            trial_index=trial_meta_data["trial_index"])

        # sync data to Ax
        potential_new_client = add_data(ax_client=ax_client, data=ax_data, trial_index=trial_meta_data["trial_index"], step=count_trials, max_building_time=cfg.objective.max_building_time)
        if potential_new_client is not None:
            current_version += 1
            ax_client_save_path = f"{save_folder}/ax_client_{current_version}.json"
            # if len(exp_to_df(ax_client.experiment)) 
            ax_client = potential_new_client
            # Save
            ax_client.save_to_json_file(filepath=ax_client_save_path)


        # log metrics and print
        log(raw_data, generation_time, count_trials)
        # Save
        if count_trials % cfg.other.save_every == 0:
            ax_client.save_to_json_file(filepath=ax_client_save_path)


    # final evaluation
    evaluate(ax_client=ax_client, step=count_trials)

@hydra.main(config_path="conf", config_name="nas", version_base="1.2")
def run_NAS(config: DictConfig) -> None:
    global cfg
    cfg = config
    #nas = NAS(cfg)
    if cfg.evaluate_only:
        pass
        #evaluate(ax_client=nas.ax_client, step=200)
    else:
        save_folder = f"NAS/data/{cfg.exp_name}"
        current_version = 0
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
        hydra_wandb_runner = init_runner(save_folder=save_folder)

            

        main_optim_loop(
            count_trials=0,
            num_trials=cfg.generation.num_total_trials,
            ax_client=ax_client,
            hydra_wandb_runner=hydra_wandb_runner,
            data_fetcher=data_fetcher,
            save_folder=save_folder,

        )


if __name__ == "__main__":
    run_NAS()