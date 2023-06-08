import re
import timeit
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
import numpy as np

from omegaconf import DictConfig
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import pprint
import sys

sys.path.append('..')
sys.path.append('../scaling-laws-ecnn') # add parent directory
from experiment.utils import allowed_usage_time, build_dataloaders
from experiment.model_instantiate import get_model
from networks.util import (
    cuda_memory_usage,
    get_param_count
)


def forward_pass(model, cfg, verbose, 
                 n_inputs, image_size, batch_size=128):
    start = timeit.default_timer()
    model.train()

    n_runs = cfg.training.steps_per_epoch

    
    for i in range(n_runs):
        input_tensor = torch.randn(batch_size, n_inputs, image_size, \
                                    image_size).cuda()
        out = model(input_tensor)
        #cuda_memory_usage()
        if cfg.other.verbose >= 3:
            print(f"Run {i}: {out.shape}, {out.sum()}")
        del out

    stop = timeit.default_timer()
    train_time = stop - start
    if verbose >= 1:
        print(f"Train time elapsed: {train_time}")


    return train_time


def main():
    # context initialization
    with initialize(config_path="../conf", version_base="1.2"):
        cfg = compose(config_name="config") #, overrides=["db=mysql", "db.user=me"])
        print(OmegaConf.to_yaml(cfg))

    n_inputs = 3
    n_outputs = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_size = cfg.dataset.resolution
    
    # instantiate model
    model, stats = get_model(
        cfg=cfg, n_inputs=n_inputs, n_outputs=n_outputs, image_size=image_size,
        device=device, verbose=cfg.other.verbose, 
    )

    # test the speed of the forward pass
    stats["train_time"] = forward_pass(model, cfg, verbose=cfg.other.verbose, 
                            n_inputs=n_inputs, image_size=image_size)

    return stats



if __name__ == "__main__":
    main()


