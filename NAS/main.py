from copy import deepcopy
import os
from pdb import run
import timeit
import hydra
import torch
import wandb
#os.environ["WANDB_SILENT"] = "true"
import json
import copy
from omegaconf import DictConfig, OmegaConf

# Ax service
from ax.service.ax_client import AxClient, ObjectiveProperties

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
from search_space_service import Eq_Search_Space
from fetch_trial_data import TrialDataFetcher
from util import init_wandb

class NAS:
    def __init__(self, cfg):
        self.cfg = cfg
        print(OmegaConf.to_yaml(cfg))
        self.save_folder = f"NAS/data/{self.cfg.exp_name}"
        self.device = torch.device('cuda' if torch.cuda.is_available() \
                                   else "cpu")
        self.json_store = {
            "ax_client": f"{self.save_folder}/ax_client.json",
            #"experiment": f"{self.save_folder}/ax_experiment.json",
            #"generation_strategy": f"{self.save_folder}/ax_generation_strategy.json",
            "wandb_run_id": f"{self.save_folder}/wandb_run_id.json",
        }
        # Generation strategy
        self.init_generation_strategy()
        # Data fetcher
        self.data_fetcher = TrialDataFetcher(
            entity=self.cfg.wandb.entity,
            project=self.cfg.wandb.project,
            wandb_mode=self.cfg.wandb.mode,
            exp_name=self.cfg.exp_name,
            max_gflops=self.cfg.objective.max_gflops,
        )
        # Ax client
        self.init_ax_client()
        # Search space
        self.init_search_space()
        # Experiment
        self.init_experiment()        
        # Runner
        self.init_runner()


    def init_ax_client(self):
        # save config
        self.wandb_config = OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=True
            )

        if not self.cfg.other.restart and os.path.exists(self.json_store["ax_client"]):
            self.ax_client = AxClient.load_from_json_file(filepath=self.json_store["ax_client"])
            # Get run id from json store
            with open(self.json_store["wandb_run_id"], 'r') as f:
                wandb_run_id = json.load(f)['wandb_run_id']
            
            # resume wandb run
            self.run = init_wandb(wandb_run_id, self.cfg)
            # connect to db
            self.data_fetcher.connect_to_db(reset=False)
        else:
            # setup ax client
            os.makedirs(self.save_folder, exist_ok=True)
            self.ax_client = AxClient(
                generation_strategy=self.generation_strategy,
            )
            # init wandb
            self.run = init_wandb(run_id=None, cfg=self.cfg, wandb_config=self.wandb_config)
            # connect to db
            self.data_fetcher.connect_to_db(reset=True)
            # Save the run_id
            with open(self.json_store["wandb_run_id"], 'w') as f:
                json.dump({'wandb_run_id': self.run.id}, f)

        
        self.run_id = self.run.id
    
    def init_experiment(self):
        self.ax_client.create_experiment(
            name=self.cfg.exp_name, 
            parameters=self.parameter,
            objectives={
                # `threshold` arguments are optional
                "valid_acc": ObjectiveProperties(
                    minimize=False, 
                    threshold=self.cfg.objective.bounds.valid_acc
                ), 
                "gflops": ObjectiveProperties(
                    minimize=True, 
                    threshold=self.cfg.objective.bounds.gflops
                )
            },
            parameter_constraints=self.parameter_constraints,
            outcome_constraints=[f"gflops <= {self.cfg.objective.max_gflops}"],
            tracking_metric_names=["train_duration", "valid_duration"],
            overwrite_existing_experiment=True,
            #is_test=True,
        )

    def init_search_space(self):
        # search space
        eq_search_space = Eq_Search_Space(
            cfg_choice_2_range_params=self.cfg.search_space.choice_2_range_params, 
            num_middle_blocks=self.cfg.search_space.num_middle_blocks
        )
        #self.search_space = eq_search_space.get_search_space()
        self.parameter = eq_search_space.get_parameters()
        self.parameter_constraints = eq_search_space.get_parameter_constraints()



    def init_runner(self):
        # we have to convert the config to a dict because the config is not serializable
        # only matters for developer api
        training_dict = OmegaConf.to_container(self.cfg.training, resolve=True)
        training_dict["NAS.max_gflops"] = self.cfg.objective.max_gflops
        training_dict["NAS.max_generation_time"] = self.cfg.objective.max_generation_time
        choice_2_range_params = OmegaConf.to_container(
            self.cfg.search_space.choice_2_range_params, resolve=True)
        
        self.hydra_wandb_runner = HydraWandbRunner(
            script_path=self.cfg.runner.script_path,
            wandb_entity=self.cfg.wandb.entity, 
            wandb_project=self.cfg.wandb.project,
            wandb_run_id=self.run_id,
            wandb_mode=self.cfg.wandb.mode_runs,
            choice_2_range_param=choice_2_range_params,
            strides=list(self.cfg.search_space.strides),
            training_dict=training_dict,
            verbose=self.cfg.runner.verbose,
        )

    
    def main_optim_loop(self):

        # Running optimization trials
        for i in range(self.cfg.generation.num_total_trials):
            if self.cfg.other.verbose >= 1:
                print(f"Trial: {i}")

            # get next trial
            start = timeit.default_timer()
            trial = self.ax_client.get_next_trial()
            print(f"trial: {trial}")
            stop = timeit.default_timer()
            generation_time = stop - start

            # run trial
            trial_meta_data =  self.hydra_wandb_runner.run(trial)

            # fetch data
            raw_data = self.data_fetcher.fetch_trial_data(
                trial_index=trial_meta_data["trial_index"])

            print(f"raw_data: {raw_data}")

            # sync data to Ax
            self.ax_client.complete_trial(
                trial_index=trial_meta_data["trial_index"], 
                raw_data=copy.deepcopy(raw_data)
            )

            # log metrics and print
            self.log(raw_data, generation_time, i)
            
            # Save
            if i % self.cfg.other.save_every == 0:
                self.ax_client.save_to_json_file(filepath=self.json_store["ax_client"])

            # Evaluate
            if i % self.cfg.other.evaluate_every == 0:
                evaluate(
                    ax_client=self.ax_client,
                    device=self.device,
                    step=i,
                )

    

    def log(self, raw_data, generation_time, step):
        raw_data["generation_time"] = generation_time
        print(f"to_log: {raw_data}")
        wandb.log(raw_data, step=step, commit=True)
        #wandb.log({"generation_time": generation_time}, step=step, commit=True)

        if self.cfg.other.verbose >= 1:
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Metrics: {raw_data}")


    def init_generation_strategy(self):
        ######################################################################
        # Choosing the Generation Strategy

        # taken from https://github.com/facebook/Ax/issues/1454
        # how to deal with large search spaces
 
        # model_fulbayes = Models.FULLYBAYESIANMOO(
        #     experiment=self.experiment, 
        #     data=data,
        #     torch_device=self.device,
        #     verbose=verbose,  # Set to True to print stats from MCMC
        #     # disable_progbar=True,  # Set to False to print a progress bar from MCMC
        # )
        # Models.SOBOL(search_space=self..search_space, seed=1234)
        self.generation_strategy=GenerationStrategy(
            name="SAASBO",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=self.cfg.generation.num_sobol_trials
                ),
                GenerationStep(
                    model=Models.FULLYBAYESIANMOO,
                    num_trials=self.cfg.generation.num_fullbayesian_trials,
                    model_kwargs={
                        "torch_device": self.device,
                        "num_samples": self.cfg.generation.num_samples,
                        "warmup_steps": self.cfg.generation.warmup_steps,
                        "disable_progbar": self.cfg.generation.progress_bar, # Set to False to print a progress bar from MCMC
                    },
                    max_parallelism=1,
                ),
            ],
        )
        # object_to_json(self.generation_strategy)
        # generation_strategy_to_json(self.json_store["generation_strategy"], self.generation_strategy)
        #save_generation_strategy(self.generation_strategy)
    
    


@hydra.main(config_path="conf", config_name="nas", version_base="1.2")
def run_NAS(cfg: DictConfig) -> None:
    nas = NAS(cfg)
    nas.main_optim_loop()


if __name__ == "__main__":
    run_NAS()