import re
import timeit
import hydra
from hydra import compose, initialize
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
from networks.util import (
    cuda_memory_usage,
    get_param_count
)

def extract_info_instantiate_network(output, verbose=False):
    """
    Extracts [param_count, model_building_time, train_time, flops] 
    from the output of the network instantiation script.
    """

    # Define the regex patterns to extract the information
    param_count_pattern = r"Total params: (\d+)"
    model_building_time_pattern = r"Model building time: ([\d.]+)"
    train_time_pattern = r"Train time elapsed: ([\d.]+)"
    flops_pattern = r"Flops: ([\d.]+)"
    # Extract the information using regex
    param_count_match = re.search(param_count_pattern, output)
    model_building_time_match = re.search(model_building_time_pattern, output)
    train_time_match = re.search(train_time_pattern, output)
    flops_match = re.search(flops_pattern, output)

    # Extracted values
    param_count = int(param_count_match.group(1)) if param_count_match else None
    model_building_time = float(model_building_time_match.group(1)) \
        if model_building_time_match else None
    train_time = float(train_time_match.group(1)) if train_time_match else None
    flops = float(flops_match.group(1)) if flops_match else None

    if verbose:
        print(f"param_count: {param_count / 1e6}M")
        print(f"model_building_time: {model_building_time}")
        print(f"train_time: {train_time}")
        print(f"FLOPs: {flops}G\n")

    return param_count, model_building_time, train_time, flops


def model_instantiate(cfg: DictConfig, verbose=1, n_inputs=3, n_outputs=10, 
                      image_size=None):
    """
    Instantiate the model and return:
    - number of parameters
    - model building time
    - train time
    - GFLOPs
    """
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

    # FLOPs
    flops = FlopCountAnalysis(model, (input_tensor,))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    gflops = flops.total() / 1e9

    if verbose >= 2:
        print(f"Flops: {gflops} GFlops")
        #print(flop_count_table(flops))
        #pprint.pprint(flops.by_operator())
        #pprint.pprint(flops.by_module())
        #pprint.pprint(flops.by_module_and_operator())
    return train_time, gflops

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(cfg)
    n_inputs = 3
    n_outputs = 10
    do_forward_pass = True
    instantiate_dataset = False

    # check if we are allowed to run
    if cfg.other.gpu_time_limit:
        allowed_usage_time()

    try: image_size = cfg.dataset.resolution
    except: image_size = 32
    
    # instantiate model
    model, param_count, model_building_time = model_instantiate(cfg, \
                            verbose=cfg.other.verbose, 
                            n_inputs=n_inputs, n_outputs=n_outputs, \
                            image_size=image_size)

    if do_forward_pass:
        train_time, gflops = forward_pass(model, cfg, verbose=cfg.other.verbose, 
                            instantiate_dataset=instantiate_dataset,
                            n_inputs=n_inputs, image_size=image_size)
    else:
        train_time = 0
    return param_count, model_building_time, train_time, gflops


if __name__ == "__main__":
    main()


