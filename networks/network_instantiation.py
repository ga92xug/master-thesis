import timeit
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import numpy as np
from omegaconf import DictConfig
import torch
#from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, flop_count_table
import pprint
import sys

sys.path.append('..')
sys.path.append('../scaling-laws-ecnn') # add parent directory
# from networks import (
#     e2wrn28_7R,
# )
from experiment.utils import allowed_usage_time, build_dataloaders
from networks.util import (
    cuda_memory_usage,
    get_param_count
)


def model_instantiate(cfg: DictConfig, verbose=1, n_inputs=3, n_outputs=10, image_size=None):
    start = timeit.default_timer()
    model = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            image_size=image_size,
        )
    stop = timeit.default_timer()
    model_building_time = stop - start
    param_count = get_param_count(model, in_mb=False, verbose=verbose)
    if verbose >= 1:
        print(f"Model building time: {model_building_time}")
    if verbose >= 3:
        print(model)

    return model, param_count, model_building_time
    
def forward_pass(model, cfg, verbose, instantiate_dataset, 
                 n_inputs, image_size, batch_size=128):
    start = timeit.default_timer()
    model.train()

    if cfg.training.steps_per_epoch > 0:
        n_runs = cfg.training.steps_per_epoch
    else:
        n_runs = 10

    if instantiate_dataset:
        dataloaders, n_inputs, n_outputs = build_dataloaders(cfg)
        for batch_idx, (x, t) in enumerate(dataloaders["train"]):
            if batch_idx >= n_runs:
                break

            input_tensor = x.cuda()
            if cfg.other.verbose >= 2:
                print(f"Run {i}: {input_tensor.shape}, {input_tensor.sum()}")
            
            out = model(input_tensor)
            # cuda memory usage
            if cfg.other.verbose >= 3:
                cuda_memory_usage()
            
            del out
    else:
        model.cuda()
        for i in range(n_runs):
            input_dim = (n_inputs, image_size, image_size)
            input_tensor = torch.randn(batch_size, n_inputs, image_size, image_size).cuda()
            out = model(input_tensor)
            #cuda_memory_usage()
            if cfg.other.verbose >= 3:
                print(f"Run {i}: {out.shape}, {out.sum()}")
            del out

    stop = timeit.default_timer()
    train_time = stop - start
    if verbose >= 1:
        print(f"Train time elapsed: {train_time}")

    if verbose >= 2:
        flops = FlopCountAnalysis(model, (input_tensor,))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        print(f"Flops: {flops.total() / 1e9} GFlops")
        print(flop_count_table(flops))
        #pprint.pprint(flops.by_operator())
        #pprint.pprint(flops.by_module())
        #pprint.pprint(flops.by_module_and_operator())
    return train_time

def run(
        overrides=[],
        instantiate_dataset = False,
        do_forward_pass = True,
        n_inputs=3, 
        n_outputs=10, 
):  
    # check if we are allowed to run
    if cfg.other.gpu_time_limit:
        allowed_usage_time()

    if isinstance(overrides, dict):
        overrides = dict_to_hydra_list(overrides)

    # print(f"Overrides: {overrides}")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base="1.2", config_path="../conf"):
        cfg = compose(config_name="config", overrides=overrides)

    try:
        image_size = cfg.dataset.resolution
    except:
        image_size = 32

    # instantiate model
    model, param_count, model_building_time = model_instantiate(cfg, verbose=cfg.other.verbose, 
                            n_inputs=n_inputs, n_outputs=n_outputs, image_size=image_size)

    if do_forward_pass:
        train_time = forward_pass(model, cfg, verbose=cfg.other.verbose, 
                            instantiate_dataset=instantiate_dataset,
                            n_inputs=n_inputs, image_size=image_size)
    else:
        train_time = 0
    return param_count, model_building_time, train_time


def dict_to_hydra_list(dict_obj):
    """
    Converts the overrides dict into a list of strings that hydra likes.
    """
    string_list = []
    for key, value in dict_obj.items():
        value_str = str(value)
        string_list.append(f"{key}={value_str}")
    return string_list

if __name__ == "__main__":
    # get arguments 
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('--overrides', type=str)
    args = parser.parse_args()
    overrides = ast.literal_eval(args.overrides)
    run(overrides=overrides)

