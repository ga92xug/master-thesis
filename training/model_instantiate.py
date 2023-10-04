import signal
from typing import List
import hydra
from hydra import compose, initialize
import timeit
import torch
from omegaconf import DictConfig
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
import sys
import os

sys.path.append(f"{os.getcwd()}")
from networks.util import get_param_count
from training.log import Log

def get_model(
        cfg: DictConfig, 
        n_inputs: int, 
        n_outputs: int, 
        image_size: int,
        device: torch.device,
        logger: Log,
        verbose: int = 1,
):
    """
    Instantiate the model and return:
    - number of parameters
    - model building time
    - train time
    - GFLOPs
    """
    is_nas = cfg.NAS.trial_index >= 0
    stats = {}
    if not is_nas:
        
        # create model
        model, model_building_time = init_model(cfg, n_inputs, n_outputs, image_size, device)

        stats["model_building_time"] = model_building_time
        stats["param_count"] = get_param_count(model, in_mb=False, verbose=verbose)
        stats["GFLOPs"] = get_gflops(model, cfg.training.dataset.batch_size, n_inputs, 
                            image_size, device=device, verbose=verbose)
    else:
        print("image_size", image_size)
        # Set the maximum allowed execution time in seconds
        max_building_time = cfg.NAS.max_building_time
        max_gflops = cfg.NAS.max_gflops

        assert max_building_time > 0, "max_building_time must be greater than 0"
        logger.log({"model_building_time": max_building_time}, step=0, epoch=0)
        assert max_gflops > 0, "max_gflops must be greater than 0"

        # Define a function to handle the timeout
        def timeout_handler(signum, frame):
            print("Model building time exceeded.")
            raise TimeoutError()

        # Set the signal handler for the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_building_time)
        
        # create model  
        model, model_building_time = init_model(cfg, n_inputs, n_outputs, image_size, device)
        # Cancel alarm
        signal.alarm(0)
        logger.log({"model_building_time": model_building_time}, step=0, epoch=0)
        
        # flops
        gflops = get_gflops(model, cfg.training.dataset.batch_size, n_inputs,
                        image_size, device=device, verbose=verbose)
        logger.log({"GFLOPs": gflops}, step=0, epoch=0)
        if gflops > max_gflops:
            raise ValueError(f"GFLOPs {gflops} exceeds maximum allowed {max_gflops}")

    ############################################################################
    # Both NAS and non-NAS

    # Compile the model
    if cfg.training.compile:
        start = timeit.default_timer()
        model = torch.compile(model)
        stop = timeit.default_timer()
        compile_time = stop - start
        stats["compile_time"] = compile_time
        
    if verbose >= 1:
        print(f"Model building time: {model_building_time}")
        if cfg.training.compile:
            print(f"Compile time: {compile_time}")
    if verbose >= 3:
        print(model)

    if logger is not None:
        logger.log(stats, step=0, epoch=0)
    return model, stats


def get_gflops(model, batch_size, n_inputs, image_size, device, verbose=False):
    input_tensor = torch.randn(batch_size, n_inputs, \
            image_size, image_size).to(device)
    flops = FlopCountAnalysis(model, (input_tensor,))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    gflops = flops.total() / 1e9
    
    if verbose >= 1:
        print(f'GFLOPs: {gflops:.2f}')
    if verbose >= 3:
        print(parameter_count_table(model))
        print(flop_count_table(flops))
    return gflops


def init_model(cfg, n_inputs, n_outputs, image_size, device):
    start = timeit.default_timer()
    model = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            image_size=image_size,
        ).to(device)
    stop = timeit.default_timer()
    model_building_time = stop - start
    return model, model_building_time
    

def test_instantiate(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    #_dataloaders, n_inputs, n_outputs = build_dataloaders(cfg)
    dataloaders, normalize_weights  = hydra.utils.call(cfg.training.dataset)
    n_inputs = cfg.training.dataset.n_in_channels
    n_outputs = cfg.training.dataset.n_out_classes
    image_size = cfg.training.dataset.resolution
    is_nas = cfg.NAS.trial_index != -1
    # model
    model, stats = get_model(
        cfg=cfg, 
        n_inputs=n_inputs, 
        n_outputs=n_outputs, 
        image_size=image_size,
        device=device,
        logger=None,
        verbose=0,
    )
    print(stats)

    if cfg.other.verbose > 5:
        print(model)

    return model, dataloaders

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def hydra_main(cfg: DictConfig) -> None:
    test_instantiate(cfg)


def hydra_compose(overrides: List[str]):
    with initialize(config_path="conf", version_base="1.2"):
        cfg = compose(config_name="config", overrides=overrides)
        model, dataloaders = test_instantiate(cfg)

    return model, dataloaders

if __name__ == "__main__":
    hydra_main()