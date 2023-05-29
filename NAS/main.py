import os
from pathlib import Path
import timeit
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from ax.core import Experiment
# Save and load
from ax.storage.sqa_store.save import save_experiment 
from ax.modelbridge.registry import Models
# Storage
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.storage.sqa_store.structs import DBSettings
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
from ax.plot.diagnostic import interact_cross_validation_plotly
# check difference between the 2
from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly

# Local
from runner import HydraWandbRunner
from search_space import Eq_Search_Space
from metric import WandbMetric
from evaluate import evaluate



"""
ToDo:
- objective_thresholds
- search space
"""

class NAS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_db()
        self.register_components()
        self.initialize_runner()
        self.init_search_space()
        self.init_generation_strategy()
        self.load_experiment()

        
    def init_search_space(self):
        # search space
        eq_search_space = Eq_Search_Space(
            choice_2_range_params=self.cfg.choice_2_range_params, 
            num_blocks=self.cfg.num_blocks
        )
        self.search_space = eq_search_space.get_search_space()

    def init_db(self):
        # Saving and loading
        init_engine_and_session_factory(url=f'sqlite:////data/{self.cfg.exp_name}.db')
        self.db_settings = DBSettings(url=f'sqlite:////data/{self.cfg.exp_name}.db')

    def register_components(self):
        # Register metric and runner classes
        register_metric(WandbMetric)
        register_runner(HydraWandbRunner)

    def load_experiment(self):
        self.run = None
        self.exp_save_path = self.cfg.other.save_path + self.cfg.exp_name + ".json"
        if not self.cfg.other.restart and os.path.exists(self.exp_save_path):
            # To load the experiment
            self.experiment = load_experiment(self.exp_save_path)

            # TODO: resume run
            
        else:
            # Creating the Experiment
            self.experiment = Experiment(
                name=self.cfg.exp_name,
                search_space=self.search_space,
                optimization_config=self.opt_config,
                runner=self.hydra_wandb_runner,
                generation_strategy=self.generation_strategy,
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

    def initialize_runner(self):
        self.hydra_wandb_runner = HydraWandbRunner(
            self.cfg.runner.script_path, 
            self.cfg.wandb.project_name_runs, 
            self.cfg.choice_2_range_params, 
            self.cfg.strides
        )

    def init_metrics(self):
        # metrics
        self.metric_val_acc = WandbMetric(
            name="valid.acc",
            entity=self.cfg.wandb.entity,
            project=self.cfg.wandb.project_name_runs,
            lower_is_better=False,
        )
        self.metric_gflops = WandbMetric(
            name="gflops",
            entity=self.cfg.wandb.entity,
            project=self.cfg.wandb.project_name_runs,
            lower_is_better=True,
        )

    def init_objective(self):
        ######################################################################
        # Setting up the ``OptimizationConfig``
        # -------------------------------------
        #
        # The way to tell Ax what it should optimize is by means of an
        # `OptimizationConfig <https://ax.dev/api/core.html#module-ax.core.optimization_config>`__.
        # Here we use a ``MultiObjectiveOptimizationConfig`` as we will
        # be performing multi-objective optimization.
        #
        # Additionally, Ax supports placing constraints on the different
        # metrics by specifying objective thresholds, which bound the region
        # of interest in the outcome space that we want to explore. For this
        # example, we will constrain the validation accuracy to be at least
        # 0.94 (94%) and the number of model parameters to be at most 80,000.
        #
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

    def init_generation_strategy(self):
        ######################################################################
        # Choosing the Generation Strategy

        # taken from https://github.com/facebook/Ax/issues/1454
        # how to deal with large search spaces
        self.generation_strategy=GenerationStrategy(
            name="SAASBO",
            steps=[
                GenerationStep(
                    model=Models.SOBOL, 
                    num_trials=self.cfg.generation.num_sobol_trials
                ),
                GenerationStep(
                    model=Models.FULLYBAYESIAN,
                    num_trials=self.cfg.generation.num_fullbayesian_trials,
                    min_trials_observed=self.cfg.generation.num_sobol_trials,
                    max_parallelism=1,
                ),
            ],
        )
    

    def evaluate(self):
        df = exp_to_df(self.experiment)
        if self.cfg.other.verbose >= 1:
            print(df)

        # Pareto frontier       
        pareto_frontier = scatter_plot_with_pareto_frontier_plotly(self.experiment)
        wandb.log({"pareto_frontier_scatter": pareto_frontier})
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
        # Running optimization trials
        for i in range(self.cfg.generation.num_total_trials):
            start = timeit.default_timer()
            trial = self.experiment.new_trial()

            stop = timeit.default_timer()
            generation_time = stop - start
            
            # Start trial run to evaluate arm(s) in the trial
            trial.run()
            trial.mark_completed()
            
            # log metrics and print
            self.log(trial, generation_time, i)

            if i % self.cfg.other.save_every == 0:
                save_experiment(self.experiment, self.exp_save_path)

            if i % self.cfg.other.evaluate_every == 0:
                evaluate(self.experiment, verbose=1)

    

    def log(self, trial, generation_time, step):
        metrics = self.metric_val_acc.get_metrics(trial)
        train_duration = metrics["train_duration"]
        valid_duration = metrics["valid_duration"]
        # Log the metrics and run_time to wandb
        self.run.log({
            "generation_time": generation_time,
            "val_acc": metrics["val_acc"], 
            "gflops": metrics["gflops"],
            "train_duration": train_duration,
            "valid_duration": valid_duration,
        }, step=step)
        self.run.log(metrics, step=step)

        if self.cfg.other.verbose >= 1:
                print(f"\nTrial: {step+1}\nGeneration time: {generation_time:.2f} seconds")
                print(f"Median train time {train_duration:.2f} seconds")

    


@hydra.main(config_path="conf", config_name="nas", version_base="1.2")
def run_NAS(cfg: DictConfig) -> None:
    nas = NAS(cfg)
    nas.main_optim_loop()


if __name__ == "__main__":
    run_NAS()