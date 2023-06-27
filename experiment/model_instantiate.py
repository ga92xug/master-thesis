import signal
import hydra
import timeit
import torch
from omegaconf import DictConfig
from fvcore.nn import FlopCountAnalysis, flop_count_table
import sys
import os

sys.path.append(f"{os.getcwd()}")
from experiment.utils import build_dataloaders
from networks.util import get_param_count
from experiment.log import Log

CUDA_CREATE_MULTIPLIER = 2.0
CUDA_CALCULATE_MULTIPLIER = 1.2


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
        stats["GFLOPs"] = get_gflops(model, cfg.training.batch_size, n_inputs, 
                            image_size, device=device, verbose=verbose)
    else:
        # Set the maximum allowed execution time in seconds
        max_building_time = cfg.NAS.max_building_time
        max_gflops = cfg.NAS.max_gflops

        assert max_building_time > 0, "max_building_time must be greater than 0"
        logger.log({"model_building_time": max_building_time}, step=0, epoch=0)
        # we currently don't restrict the gflops
        # assert max_gflops > 0, "max_gflops must be greater than 0"

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
        gflops = get_gflops(model, cfg.training.batch_size, n_inputs,
                        image_size, device=device, verbose=verbose)
        logger.log({"GFLOPs": gflops}, step=0, epoch=0)

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
    

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def test_instantiate(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    _dataloaders, n_inputs, n_outputs = build_dataloaders(cfg)
    is_nas = cfg.NAS.trial_index != -1
    # model
    model, stats = get_model(
        cfg=cfg, 
        n_inputs=n_inputs, 
        n_outputs=n_outputs, 
        image_size=cfg.training.dataset.resolution,
        device=device,
        logger=None,
        verbose=0,
    )
    print(stats)


if __name__ == "__main__":
    test_instantiate()