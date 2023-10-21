import numpy as np
from ray.tune.stopper.stopper import Stopper


class HPOEarlyStopper(Stopper):
    def __init__(self, mode, metric, patience, min_num_trials, min_delta):
        self.mode = mode
        self.metric = metric
        self.patience = patience
        self.min_num_trials = min_num_trials
        self.min_delta = min_delta
        self.no_improve_counter = 0
        self.best_score = None
        self.num_trials = 0

    def __call__(self, trial_id, result):
        self.num_trials += 1
        current_score = result[self.metric]

        # Record the best score irrespective of the number of trials
        if self.best_score is None:
            self.best_score = current_score

        if self.mode == "min":
            better_score = current_score < self.best_score - self.min_delta
        else:  # self.mode == "max"
            better_score = current_score > self.best_score + self.min_delta

        if better_score:
            self.best_score = current_score

        # Only start counting the number of "non-improvements" after min_num_trials
        if self.num_trials >= self.min_num_trials and not better_score:
            self.no_improve_counter += 1

    def stop_all(self) -> bool:
        if self.no_improve_counter >= self.patience and \
            self.num_trials >= self.min_num_trials:
            return True

        return False
