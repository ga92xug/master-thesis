from calendar import c
import os
from pathlib import Path
from random import choice
import timeit
import hydra
import torch
import wandb
import json
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

class NAS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() \
                                   else "cpu")

        self.ax_client = AxClient()
        
        # saving
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


    def init_db(self):
        # Saving and loading
        init_engine_and_session_factory(url=f'sqlite:///{self.save_folder}/ax_experiment.db')
        self.db_settings = DBSettings(url=f'sqlite:///{self.save_folder}/ax_experiment.db')
        engine = get_engine()
        create_all_tables(engine)


    def register_components(self):
        # Register metric and runner classes
        bundle = RegistryBundle(
                metric_clss={WandbMetric: None},
                runner_clss={HydraWandbRunner: None}
            )
        
        self.sqa_config = SQAConfig(
            json_encoder_registry=bundle.encoder_registry,
            json_decoder_registry=bundle.decoder_registry,
            metric_registry=bundle.metric_registry,
            runner_registry=bundle.runner_registry,
        )


    def init_search_space(self):
        # search space
        eq_search_space = Eq_Search_Space(
            cfg_choice_2_range_params=self.cfg.search_space.choice_2_range_params, 
            num_middle_blocks=self.cfg.search_space.num_middle_blocks
        )
        self.search_space = eq_search_space.get_search_space()


    def init_experiment(self):
        self.run = None
        self.json_store = {
            # currently we save experiment and generation strategy with sqa
            # "experiment": f"{self.save_folder}/ax_experiment.json",
            # "generation_strategy": f"{self.save_folder}/ax_generation_strategy.json",
            "wandb_run_id": f"{self.save_folder}/wandb_run_id.json",
        }

        if not self.cfg.other.restart and os.path.exists(self.exp_save_path):
            # To load the experiment
            self.experiment = load_experiment(self.cfg.exp_name, 
                                              config=self.sqa_config)
            # To load the generation strategy
            self.generation_strategy = load_generation_strategy_by_experiment_name(
                experiment_name=self.cfg.exp_name,
                config=self.sqa_config,
            )

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
            # Creating the Experiment
            self.experiment = Experiment(
                name=self.cfg.exp_name,
                search_space=self.search_space,
                optimization_config=self.opt_config,
                runner=self.hydra_wandb_runner,
                tracking_metrics=[self.metric_val_acc, self.metric_gflops]
            )

            # init wandb run
            wandb_config = OmegaConf.to_container(
                    self.cfg, resolve=True, throw_on_missing=True
                )
            self.run = wandb.init(
                project=self.cfg.wandb.high_level.project, 
                entity=self.cfg.wandb.entity, 
                mode=self.cfg.wandb.high_level.mode,
                config=wandb_config,
            )

            # Save the experiment
            self.save_all(save_wandb_run_id=True)

    def save_all(self, save_wandb_run_id=False):
        try:
            # Save the experiment
            save_experiment(self.experiment, config=self.sqa_config)
        except:
            # Delete the experiment since there does not seem to be a way to update it
            delete_experiment(self.cfg.exp_name)
            # Save the experiment
            save_experiment(self.experiment, config=self.sqa_config)

        save_generation_strategy(self.generation_strategy, 
                                     config=self.sqa_config)

        # Save the run_id
        if save_wandb_run_id:
            with open(self.json_store["wandb_run_id"], 'w') as f:
                json.dump({'wandb_run_id': self.run.id}, f)


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
        self.metric_gflops = WandbMetric(
            name="gflops",
            entity=self.cfg.wandb.entity,
            project=self.cfg.wandb.runs.project,
            wandb_mode=self.cfg.wandb.runs.mode,
            lower_is_better=True,
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


    def init_objective(self):
        self.opt_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=self.metric_val_acc, minimize=False),
                    Objective(metric=self.metric_gflops, minimize=True),
                ],
            ),
            objective_thresholds=[
                ObjectiveThreshold(
                    metric=self.metric_val_acc, 
                    bound=self.cfg.objective.bounds.val_acc, 
                    relative=False
                ),
                ObjectiveThreshold(
                    metric=self.metric_gflops, 
                    bound=self.cfg.objective.bounds.gflops, 
                    relative=False
                ),
            ],
        )

    def initialize_experiment(self):
        sobol = Models.SOBOL(search_space=self.search_space, seed=self.cfg.seed)

        for _ in range(self.cfg.generation.num_sobol_trials):
            self.experiment.new_trial(sobol.gen(1)).run()

        return self.experiment.fetch_data()
    
    def get_model(self, data):
        pass


    def evaluate(self):
        df = exp_to_df(self.experiment)
        if self.cfg.other.verbose >= 1:
            print(df)

        # Pareto frontier       
        #pareto_frontier = scatter_plot_with_pareto_frontier_plotly(self.experiment)
        #wandb.log({"pareto_frontier_scatter": pareto_frontier})
        pareto_frontier = _pareto_frontier_scatter_2d_plotly(self.experiment)
        wandb.log({"pareto_frontier_scatter": pareto_frontier})

        # Plotting the model fit
        model = self.generation_strategy.model
        cv = cross_validate(model=model)  
        compute_diagnostics(cv)
        cross_validation = interact_cross_validation_plotly(cv)
        wandb.log({"cross_validation": cross_validation})
        
        # Plotting the optimization goals against the search space
        val_contour_plot = interact_contour_plotly(model=model, metric_name="val_acc")
        gflops_contour_plot = interact_contour_plotly(model=model, metric_name="gflops")
        wandb.log({"val_contour_plot": val_contour_plot,
                    "gflops_contour_plot": gflops_contour_plot})
        

    
    def main_optim_loop(self):
        from ax.service.scheduler import Scheduler, SchedulerOptions


        scheduler = Scheduler(
            experiment=self.experiment,
            generation_strategy=self.generation_strategy,
            options=SchedulerOptions(),
        )
        print("Use the scheduler to run the optimization loop.")
        scheduler.run_n_trials(max_trials=1)
        print(scheduler.experiment.trials.values())

        # Running optimization trials
        for i in range(self.cfg.generation.num_total_trials):
            start = timeit.default_timer()

            generator_run = self.generation_strategy.gen(
                experiment=self.experiment, 
                # data=None,  
                n=1, # Number of candidate arms to produce
            )

            trial = self.experiment.new_trial(generator_run)

            stop = timeit.default_timer()
            generation_time = stop - start
            
            # Start trial run to evaluate arm(s) in the trial
            trial.run()
            trial.mark_completed()
            print(f"Trial {i} completed")
            trial.fetch_data()
            # Evaluate the trial and add the results to the trial
            #results = self.fetch_trial_data(trial)
            #for metric_name, metric_value in results.items():
            #    trial.add_run_data(metric_name=metric_name, mean=metric_value)
# 
            # self.experiment.attach_trial(trial)

            # self.experiment.fetch_data()
            #data = Data.from_multiple_data([data, trial.fetch_data()])
            
            # log metrics and print
            self.log(trial, generation_time, i)

            if i % self.cfg.other.save_every == 0:
                self.save_all(save_wandb_run_id=False)

            if i % self.cfg.other.evaluate_every == 0:
                self.evaluate()

    

    def log(self, trial, generation_time, step):
        metrics = self.metric_val_acc.get_metrics(trial)
        train_duration = metrics["train_duration"]
        valid_duration = metrics["valid_duration"]
        # Log the metrics and run_time to wandb
        self.run.log({
            "generation_time": generation_time,
            "valid_acc": metrics["valid.acc"], 
            "gflops": metrics["gflops"],
            "train_duration": train_duration,
            "valid_duration": valid_duration,
        }, step=step)

        if self.cfg.other.verbose >= 1:
            print(f"\nTrial: {step+1}\nGeneration time: {generation_time:.2f} seconds")
            print(f"Median train time {train_duration:.2f} seconds")




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
        save_generation_strategy(self.generation_strategy)
    
    


@hydra.main(config_path="conf", config_name="nas", version_base="1.2")
def run_NAS(cfg: DictConfig) -> None:
    nas = NAS(cfg)
    nas.main_optim_loop()


if __name__ == "__main__":
    run_NAS()