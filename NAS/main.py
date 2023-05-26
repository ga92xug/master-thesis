from pathlib import Path
from ax.core import Experiment
# Save and load
from ax.service.utils.instantiation import save_experiment, load_experiment
from ax.modelbridge.registry import Models
# Storage
from ax.storage.sqlalchemy_store.db import init_engine_and_session_factory
from ax.storage.sqlalchemy_store.load import load_experiment
from ax.storage.sqlalchemy_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.storage import DBSettings
# Objective
from ax.core import MultiObjective, Objective, ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
# Generation Strategy
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy

# Local
from runner import HydraWandbRunner
from search_space import Eq_Search_Space
from metric import WandbMetric
from evaluate import evaluate

EXP_NAME = "mnist_rot"
TOTAL_TRIALS = 48  # total evaluation budget
NUM_SOBOL_TRIALS = 5
NUM_BOTORCH_TRIALS = 15


"""
ToDo:
- objective_thresholds
- search space

"""

######################################################################
# Saving and loading
init_engine_and_session_factory(url=f'sqlite:////data/{EXP_NAME}.db')
db_settings = DBSettings(url=f'sqlite:////data/{EXP_NAME}.db')
# Register metric and runner classes
register_metric(WandbMetric)
register_runner(HydraWandbRunner)

######################################################################
# runner
project_name = "scaling-laws-eq"
script_path = "experiment/main.py"  
hydra_wandb_runner = HydraWandbRunner(script_path, project_name)
#runner.run(trial)

######################################################################
# search space
eq_search_space = Eq_Search_Space.Eq_Search_Space()
search_space = eq_search_space.get_search_space()

######################################################################
# metrics
metric_val_acc = WandbMetric(
    name="valid.acc",
    lower_is_better=False,
)
metric_gflops = WandbMetric(
    name="gflops",
    lower_is_better=True,
)


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

opt_config = MultiObjectiveOptimizationConfig(
    objective=MultiObjective(
        objectives=[
            Objective(metric=metric_val_acc, minimize=False),
            Objective(metric=metric_gflops, minimize=True),
        ],
    ),
    objective_thresholds=[
        ObjectiveThreshold(metric=metric_val_acc, bound=0.94, relative=False),
        ObjectiveThreshold(metric=metric_gflops, bound=80_000, relative=False),
    ],
)

######################################################################
# Creating the Experiment
experiment = Experiment(
    name=EXP_NAME,
    search_space=search_space,
    optimization_config=opt_config,
    runner=hydra_wandb_runner,
)

######################################################################
# Choosing the Generation Strategy
gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials=TOTAL_TRIALS,
  )


# taken from https://github.com/facebook/Ax/issues/1454
# how to deal with large search spaces
generation_strategy=GenerationStrategy(
    name="SAASBO",
    steps=[
        GenerationStep(model=Models.SOBOL, num_trials=10),
        GenerationStep(
            model=Models.FULLYBAYESIAN,
            num_trials=-1,
            min_trials_observed=10,
            max_parallelism=1,
        ),
    ],
)

######################################################################
# Running optimization trials
LOCATION_PATH = "data/"
EXP_SAVE_PATH = LOCATION_PATH + EXP_NAME + ".json"

print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(search_space=experiment.search_space)
    
for i in range(NUM_SOBOL_TRIALS):
    # Produce a GeneratorRun from the model, which contains proposed arm(s) and other metadata
    generator_run = sobol.gen(n=1)
    # Add generator run to a trial to make it part of the experiment and evaluate arm(s) in it
    trial = experiment.new_trial(generator_run=generator_run)
    # Start trial run to evaluate arm(s) in the trial
    trial.run()
    trial.mark_completed()
    # Save the experiment after each trial
    save_experiment(experiment, EXP_SAVE_PATH)

# To load the experiment
#loaded_exp = load_experiment(EXP_SAVE_PATH)


evaluate(experiment, verbose=1)