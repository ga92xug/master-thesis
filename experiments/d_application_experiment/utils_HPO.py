from ray.tune.stopper import Stopper

class HPOEarlyStopper(Stopper):
    def __init__(self, mode, metric, patience, min_delta: float = 0.001):
        self.mode = mode
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.global_best_score = None
        self.no_improve_counter = 0
        self.seen_trial_ids = set()

    def __call__(self, trial_id, result):
        current_score = result[self.metric]

        # Always update the global best score, irrespective of trial_id
        if self.mode == "max" and (self.global_best_score is None or current_score > self.global_best_score + self.min_delta):
            self.global_best_score = current_score
            self.no_improve_counter = 0  # reset counter
        elif self.mode == "min" and (self.global_best_score is None or current_score < self.global_best_score - self.min_delta):
            self.global_best_score = current_score
            self.no_improve_counter = 0  # reset counter

        # Increment no_improve_counter only for new trial_ids
        if trial_id not in self.seen_trial_ids:
            self.seen_trial_ids.add(trial_id)
            self.no_improve_counter += 1

    def stop_all(self) -> bool:
        return self.no_improve_counter >= self.patience
