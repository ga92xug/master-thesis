from calendar import c
from copy import deepcopy
import datetime
import os
from pathlib import Path
from random import choice
import timeit
import hydra
import torch
import wandb
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
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.service.utils.report_utils import exp_to_df
from ax.plot.pareto_frontier import scatter_plot_with_pareto_frontier_plotly
from ax.plot.contour import interact_contour_plotly
from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate
# check difference between the 2
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly
# Models
from ax.modelbridge.registry import Models

# Local
from runner import HydraWandbRunner
from search_space import Eq_Search_Space
from metric import WandbMetric
from plot import plot_pareto_frontier

class NAS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_folder = f"NAS/data/{self.cfg.exp_name}"
        self.json_store = {
            # currently we save experiment and generation strategy with sqa
            "ax_client": f"{self.save_folder}/ax_client.json",
            #"experiment": f"{self.save_folder}/ax_experiment.json",
            #"generation_strategy": f"{self.save_folder}/ax_generation_strategy.json",
            "wandb_run_id": f"{self.save_folder}/wandb_run_id.json",
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() \
                                   else "cpu")
        self.init_generation_strategy()
        #self.init_db()
        self.init_ax_client()
        
        self.init_search_space()
        print(self.parameter)
        # get timestamp
        
        self.ax_client.create_experiment(
            name=self.cfg.exp_name + str(datetime.datetime.now()),
            parameters=self.parameter,
            objectives={
                # `threshold` arguments are optional
                "val_acc": ObjectiveProperties(minimize=False, threshold=self.cfg.objective.bounds.val_acc), 
                "gflops": ObjectiveProperties(minimize=True, threshold=self.cfg.objective.bounds.gflops)
            },
            parameter_constraints=self.parameter_constraints,
            #outcome_constraints=["valid_acc >= 0.9"],
            tracking_metric_names=["train_duration", "val_duration"],
            overwrite_existing_experiment=True,
            #is_test=True,
        )
        self.init_metrics()
        self.init_runner()


        # saving
        """
        self.save_folder = f"NAS/data/{self.cfg.exp_name}"
        os.makedirs(self.save_folder, exist_ok=True)
        self.init_db()
        self.register_components()

        # setup experiment
        self.init_metrics()
        self.init_runner()
        self.init_search_space()
        self.init_generation_strategy()
        self.init_objective()
        self.init_experiment()
        """

    def init_ax_client(self):
        # save config
        wandb_config = OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=True
            )

        if not self.cfg.other.restart and os.path.exists(self.json_store["ax_client"]):
            self.ax_client = AxClient.load_from_json_file(file_path=self.json_store["ax_client"])
            # Get run id from json store
            with open(self.json_store["wandb_run_id"], 'r') as f:
                wandb_run_id = json.load(f)['wandb_run_id']
            # resume wandb run
            self.run = wandb.init(
                project=self.cfg.wandb.high_level.project, 
                entity=self.cfg.wandb.entity, 
                mode=self.cfg.wandb.high_level.mode,
                config=wandb_config,
                resume="allow",
                id=wandb_run_id,  # resume the run using the saved run ID
            )

        else:
            # setup ax client
            os.makedirs(self.save_folder, exist_ok=True)
            self.ax_client = AxClient(
                generation_strategy=self.generation_strategy,
                #db_settings=self.db_settings,
                torch_device=self.device,
            )
            self.run = wandb.init(
                project=self.cfg.wandb.high_level.project, 
                entity=self.cfg.wandb.entity, 
                mode=self.cfg.wandb.high_level.mode,
                config=wandb_config,
            )
            # Save the run_id
            with open(self.json_store["wandb_run_id"], 'w') as f:
                json.dump({'wandb_run_id': self.run.id}, f)
            #self.ax_client.load_experiment_from_json(self.json_store["experiment"])
            #self.ax_client.load_generation_strategy_from_json(self.json_store["generation_strategy"])

    def init_db(self):
        # Saving and loading
        init_engine_and_session_factory(url=f'sqlite:///{self.save_folder}/ax_experiment.db')
        self.db_settings = DBSettings(url=f'sqlite:///{self.save_folder}/ax_experiment.db')
        engine = get_engine()
        create_all_tables(engine)


    

    def init_search_space(self):
        # search space
        eq_search_space = Eq_Search_Space(
            cfg_choice_2_range_params=self.cfg.search_space.choice_2_range_params, 
            num_middle_blocks=self.cfg.search_space.num_middle_blocks
        )
        #self.search_space = eq_search_space.get_search_space()
        self.parameter = eq_search_space.get_parameters()
        self.parameter_constraints = eq_search_space.get_parameter_constraints()


    def init_metrics(self):
        # metrics
        self.metric_val_acc = WandbMetric(
            name="valid.acc",
            entity=self.cfg.wandb.entity,
            project=self.cfg.wandb.runs.project,
            wandb_mode=self.cfg.wandb.runs.mode,
            lower_is_better=False,
            exp_name=self.cfg.exp_name,
        )


    def init_runner(self):
        training_dict = OmegaConf.to_container(self.cfg.training, resolve=True)
        choice_2_range_params = OmegaConf.to_container(
            self.cfg.search_space.choice_2_range_params, resolve=True)
        self.hydra_wandb_runner = HydraWandbRunner(
            script_path=self.cfg.runner.script_path,
            wandb_entity=self.cfg.wandb.entity, 
            wandb_project=self.cfg.wandb.runs.project,
            wandb_mode=self.cfg.wandb.runs.mode,
            choice_2_range_param=choice_2_range_params,
            strides=list(self.cfg.search_space.strides),
            training_dict=training_dict,
            verbose=self.cfg.runner.verbose,
        )


    def evaluate(self, i):
        df = exp_to_df(self.ax_client.experiment)
        if self.cfg.other.verbose >= 1:
            print(df)

        #objectives = self.ax_client.experiment.optimization_config.objective.objectives
        #frontier = compute_posterior_pareto_frontier(
        #    experiment=self.ax_client.experiment,
        #    data=self.ax_client.experiment.fetch_data(),
        #    primary_objective=objectives[0].metric,
        #    secondary_objective=objectives[1].metric,
        #    absolute_metrics=["val_acc", "gflops"],
        #    #num_points=20,
        #)
        #pareto_frontier = plot_pareto_frontier(frontier, CI_level=0.90)

        # Pareto frontier       
        pareto_frontier = scatter_plot_with_pareto_frontier_plotly(self.ax_client.experiment)
        #wandb.log({"pareto_frontier_scatter": pareto_frontier})
        #pareto_frontier = _pareto_frontier_scatter_2d_plotly(self.experiment)
        wandb.log({"pareto_frontier_scatter": pareto_frontier})

        """
        # Plotting the model fit
        model = self.generation_strategy.model
        cv = cross_validate(model=model)  
        compute_diagnostics(cv)
        cross_validation = interact_cross_validation_plotly(cv)
        wandb.log({"cross_validation": cross_validation})
        """
        which_model = 0 if i < self.cfg.generation.num_sobol_trials else -1
        # Plotting the optimization goals against the search space
        val_contour_plot = interact_contour_plotly(model=self.ax_client.generation_strategy[0], metric_name="val_acc")
        gflops_contour_plot = interact_contour_plotly(model=self.ax_client.generation_strategy[0], metric_name="gflops")
        wandb.log({"val_contour_plot": val_contour_plot,
                    "gflops_contour_plot": gflops_contour_plot})
        

    
    def main_optim_loop(self):

        # Running optimization trials
        for i in range(self.cfg.generation.num_total_trials):
            start = timeit.default_timer()

            
            # parameters, trial_index
            trial = self.ax_client.get_next_trial()
            trial_meta_data =  self.hydra_wandb_runner.run(trial)
            parameters, trial_index = trial
            raw_data = self.metric_val_acc.fetch_trial_data(trial_index=trial_index, wandb_run_id=trial_meta_data["wandb_run_id"])
            input_exp = copy.deepcopy(raw_data)
            # Local evaluation here can be replaced with deployment to external system.
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=input_exp)
            print(f"Trial {i} completed")
            print(f"raw_data: {raw_data}")

            stop = timeit.default_timer()
            generation_time = stop - start

            # log metrics and print
            self.log(raw_data, generation_time, i)

            if i % self.cfg.other.save_every == 0:
                self.ax_client.save_to_json_file(filepath=self.json_store["ax_client"])

            if i % self.cfg.other.evaluate_every == 0:
                self.evaluate(i)

    

    def log(self, raw_data, generation_time, step):
        #metrics = self.metric_val_acc.get_metrics(trial)
        raw_data["generation_time"] = generation_time
        # Log the metrics and run_time to wandb
        self.run.log(raw_data, step=step)

        if self.cfg.other.verbose >= 1:
            print(f"\nTrial: {step+1}\nGeneration time: {generation_time:.2f} seconds")
            print(f"Metrics: {raw_data}")




    def init_generation_strategy(self):
        ######################################################################
        # Choosing the Generation Strategy

        # taken from https://github.com/facebook/Ax/issues/1454
        # how to deal with large search spaces
        # verbose = self.cfg.other.verbose >= 2
# 
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