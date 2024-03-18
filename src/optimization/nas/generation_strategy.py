from omegaconf import DictConfig
import torch
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

def init_generation_strategy(
        cfg: DictConfig,
    ) -> GenerationStrategy:
    ######################################################################
    # Choosing the Generation Strategy
    # taken from https://github.com/facebook/Ax/issues/1454
    # how to deal with large search spaces
    device = torch.device('cuda' if torch.cuda.is_available() \
                               else "cpu")
    steps = []
    num_sobol_trials = cfg.generation.num_sobol_trials
    if num_sobol_trials > 0:
        steps.append(
            GenerationStep(
                model=Models.SOBOL,
                num_trials=cfg.generation.num_sobol_trials,
                should_deduplicate=True,
            )
        )
    steps.append(
        GenerationStep(
                model=Models.FULLYBAYESIANMOO,
                num_trials=cfg.generation.num_fullbayesian_trials,
                model_kwargs={
                    "torch_device": device,
                    "num_samples": cfg.generation.num_samples,
                    "warmup_steps": cfg.generation.warmup_steps,
                    "disable_progbar": cfg.generation.disable_progbar, # Set to False to print a progress bar from MCMC
                },
                max_parallelism=1,
                should_deduplicate=True,
            )
    )
    generation_strategy=GenerationStrategy(
        name="SAASBO",
        steps=steps,
    )
    return generation_strategy