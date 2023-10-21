import numpy as np
import ray
from ray import train, tune
from ray.tune.stopper.stopper import Stopper
from ray.tune.stopper import ExperimentPlateauStopper
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.ax import AxSearch

import os
import sys
sys.path.append(f"{os.getcwd()}")
#os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
from training.main import hydra_initialize_init


def run_HPO(
        name: str,
        search_space: dict, 
        additional_overrides: dict,
        optimize_for: str,
        optimize_mode: str,
        grace_period: int,
        num_trials: int,
        restore: bool = False,
        debug: bool = False,
    ):
    print("Search space:", search_space)
    print("Additional overrides:", additional_overrides)

    trainable_with_resources = tune.with_resources(hydra_initialize_init, {"gpu": 1})
    trainable_with_parameters = tune.with_parameters(trainable_with_resources, 
        additional_overrides=additional_overrides)

    algo = AxSearch()
    asha_scheduler = ASHAScheduler(grace_period=grace_period)

    tune_config=tune.TuneConfig(
        metric=optimize_for,
        mode=optimize_mode,
        search_alg=algo,
        scheduler=asha_scheduler,
        num_samples=num_trials,
    )

    run_config=train.RunConfig(name=name)

    stopper = ExperimentPlateauStopper(
        metric=optimize_for, 
        std=0.001,
        top=5,
        mode=optimize_mode,  
        patience=3,
        # min number of trials before stopping
        min_trials=max(num_trials//2, 15)
    )
    
    if restore:
        tuner = tune.Tuner.restore(
            os.path.expanduser(f"~/ray_results/{name}"),
            trainable=trainable_with_parameters,
            resume_unfinished=True,
            resume_errored=True,
        )
    else:
        # remove old results
        if os.path.exists(os.path.expanduser(f"~/ray_results/{name}")):
            os.system(f"rm -rf ~/ray_results/{name}")
        tuner = tune.Tuner(
            trainable_with_parameters,
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )
    
    if debug:
        print("Debugging...")
        return

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
    


class ExperimentPlateauStopper(Stopper):
    """Early stop the experiment when a metric plateaued across trials.

    Stops the entire experiment when the metric has plateaued
    for more than the given amount of iterations specified in
    the patience parameter.

    Args:
        metric: The metric to be monitored.
        std: The minimal standard deviation after which
            the tuning process has to stop.
        top: The number of best models to consider.
        mode: The mode to select the top results.
            Can either be "min" or "max".
        patience: Number of epochs to wait for
            a change in the top models.

    Raises:
        ValueError: If the mode parameter is not "min" nor "max".
        ValueError: If the top parameter is not an integer
            greater than 1.
        ValueError: If the standard deviation parameter is not
            a strictly positive float.
        ValueError: If the patience parameter is not
            a strictly positive integer.
    """

    def __init__(
        self,
        metric: str,
        std: float = 0.001,
        top: int = 10,
        mode: str = "min",
        patience: int = 0,
        min_trials: int = 5  # New parameter
    ):
        if mode not in ("min", "max"):
            raise ValueError("The mode parameter can only be either min or max.")
        if not isinstance(top, int) or top <= 1:
            raise ValueError(
                "Top results to consider must be"
                " a positive integer greater than one."
            )
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("Patience must be a strictly positive integer.")
        if not isinstance(std, float) or std <= 0:
            raise ValueError(
                "The standard deviation must be a strictly positive float number."
            )
        self._mode = mode
        self._metric = metric
        self._patience = patience
        self._iterations = 0
        self._std = std
        self._top = top
        self.values = []
        self._min_trials = min_trials
        self._trial_count = 0  # Initialize trial count

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        self._trial_count += 1  # Increment trial count

        self._top_values.append(result[self._metric])
        if self._mode == "min":
            self._top_values = sorted(self._top_values)[: self._top]
        else:
            self._top_values = sorted(self._top_values)[-self._top :]

        # If the current iteration has to stop
        if self.has_plateaued():
            # we increment the total counter of iterations
            self._iterations += 1
        else:
            # otherwise we reset the counter
            self._iterations = 0

        # and then call the method that re-executes
        # the checks, including the iterations.
        return self.stop_all()

    def has_plateaued(self):
        return (
            len(self._top_values) == self._top and np.std(self._top_values) <= self._std
        )

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return (self._trial_count >= self._min_trials) and self.has_plateaued() and self._iterations >= self._patience


"""
0.762499988079071
0.759999990463257
0.730000019073486
0.75
0.774999976158142
0.75
0.774999976158142
0.740000009536743
0.725000023841858
0.759999990463257
0.774999976158142
0.774999976158142
0.774999976158142
0.774999976158142
0.762499988079071
0.774999976158142
0.774999976158142
0.774999976158142
0.774999976158142
0.764999985694885
0.759999990463257
0.774999976158142
0.759999990463257
0.774999976158142
0.774999976158142
0.767499983310699
0.774999976158142
0.75
0.757499992847443
0.757499992847443
0.759999990463257
0.752499997615814
0.752499997615814
0.747500002384186
0.754999995231628
0.757499992847443
0.737500011920929
0.754999995231628
0.762499988079071
0.759999990463257
0.732500016689301
0.769999980926514
0.685000002384186
0.707499980926514
0.742500007152557
0.670000016689301
0.702499985694885
0.742500007152557
0.675000011920929
0.714999973773956
0.725000023841858
0.567499995231628
"""