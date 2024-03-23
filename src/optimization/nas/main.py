import os
import shutil
import random
import timeit
from typing import Dict
import hydra
import torch
import wandb
from omegaconf import DictConfig
import logging
from ax.utils.common.logger import ROOT_STREAM_HANDLER
ROOT_STREAM_HANDLER.setLevel(logging.ERROR)
# Local
import os
import sys
sys.path.append(os.getcwd())
from src.optimization.nas.client import AxClientWrapper
from src.optimization.nas.submit_jobs import Hydra_Submitter
from src.optimization.nas.data_fetcher import fetch_trial_data


def optimization_loop(
        cfg: DictConfig,
    ):
    ax_client = AxClientWrapper(cfg=cfg)
    submitor = Hydra_Submitter(additional_overrides=cfg.dataset.additional_overrides)

    count_trials = 0
    num_trials = cfg.generation.num_total_trials

    # Running optimization trials
    while count_trials < num_trials:
        # get next trial
        trial = ax_client.get_next_trial()

        # run trial
        trial_meta_data = submitor.run(trial)
        trial_index=trial_meta_data["trial_index"]

        # fetch data
        data = fetch_trial_data(
            trial_index=trial_index,
            base_path=cfg.save_path,
            metric_name=cfg.dataset.objective.objectives._0.name,
        )

        # sync data to Ax
        ax_client.add_data(
            data=data, 
            trial_index=trial_index, 
        )

        # Save
        if count_trials % cfg.other.save_every == 0:
            ax_client.save(count_trials)
        
        count_trials += 1

    # final evaluation
    ax_client.save(count_trials)


@hydra.main(config_path="configs", config_name="conf", version_base="1.3")
def main(cfg: DictConfig) -> None:
    optimization_loop(cfg)


if __name__ == "__main__":
    main()