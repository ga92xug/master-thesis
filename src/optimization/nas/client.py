import os
from typing import Dict
from omegaconf import DictConfig

# Ax service
from ax.service.ax_client import AxClient, ObjectiveProperties

import logging
from ax.utils.common.logger import ROOT_STREAM_HANDLER
ROOT_STREAM_HANDLER.setLevel(logging.ERROR)

# Local
from src.optimization.nas.generation_strategy import init_generation_strategy
from src.optimization.nas.search_space import Search_Space

class AxClientWrapper:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.max_building_time = cfg.dataset.objective.max_building_time
        self.client_save_path = os.path.join(cfg.save_path, "ax_client.json")
        os.makedirs(os.path.dirname(self.client_save_path), exist_ok=True)

        # Generation strategy
        generation_strategy = init_generation_strategy(cfg)
        # setup ax client
        self.ax_client = AxClient(
            generation_strategy=generation_strategy,
            random_seed=cfg.seed,
        )
        # Search space
        self.eq_search_space = Search_Space(search_space_cfg=cfg.dataset.search_space)
        self.parameter = self.eq_search_space.get_parameters()
        self.parameter_constraints = self.eq_search_space.get_parameter_constraints()

        # create experiment
        self.create_experiment()

    def create_objective(self) -> Dict[str, ObjectiveProperties]:
        objectives = {}
        for objective_index, objective_values in self.cfg.dataset.objective.objectives.items():
            objectives[objective_values.name] = ObjectiveProperties(
                minimize=objective_values.minimize,
            )
        return objectives

    def create_experiment(self):
        # new experiment
        self.ax_client.create_experiment(
            name=self.cfg.exp_name, 
            parameters=self.parameter,
            support_intermediate_data=True,
            objectives=self.create_objective(),
            parameter_constraints=self.parameter_constraints,
            outcome_constraints=[f"net_building_time <= {self.max_building_time}"],
            tracking_metric_names=["net_building_time"],
            overwrite_existing_experiment=True,
        )      

    def add_data(
        self,
        data: Dict[str, float], 
        trial_index: int, 
        ) -> None:

        if len(data) == 0:
            raise ValueError("Data is empty")
        elif len(data) in [1, 2]:
            if data["net_building_time"] < self.max_building_time:
                # this is not expected
                #data["net_building_time"] = max_building_time
                raise ValueError("Data is not complete", data, self.max_building_time)
            
            # early stop trial 
            # expected if the model building time exceeds the limit
            # or flops are to large
            self.ax_client.update_running_trial_with_intermediate_data(
                trial_index=trial_index,
                raw_data=data,
            )
            self.ax_client.stop_trial_early(
                trial_index=trial_index,
            )
        else:
            # complete trial
            self.ax_client.complete_trial(
                trial_index=trial_index, 
                raw_data=data,
            )

    def save(self, count_trials: int = -1) -> None:
        if count_trials == -1:
            self.ax_client.save_to_json_file(filepath=self.client_save_path)     
        elif count_trials % self.cfg.save_every == 0:
            self.ax_client.save_to_json_file(filepath=self.client_save_path)

    def get_next_trial(self):
        return self.ax_client.get_next_trial()

