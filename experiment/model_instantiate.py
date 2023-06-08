import signal
import hydra
import timeit
import torch
from omegaconf import DictConfig
from fvcore.nn import FlopCountAnalysis, flop_count_table
from networks.util import get_param_count
from log import Log


TIMEOUT_MULTIPLIER = 1.5
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
        max_building_time = cfg.nas.max_building_time
        max_gflops = cfg.NAS.max_gflops

        assert max_building_time > 0, "max_building_time must be greater than 0"
        assert max_gflops > 0, "max_gflops must be greater than 0"

        # Set the signal handler for the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_building_time)

        # create model
        try:
            model, model_building_time = init_model(cfg, n_inputs, n_outputs, image_size, device)
            # Cancel the alarm since the command finished before the timeout
            signal.alarm(0)
        except TimeoutError as e:
            # Handle the timeout
            stats["GFLOPs"] = int(max_gflops * TIMEOUT_MULTIPLIER)
            logger.log(stats, step=0, epoch=0)
            raise RuntimeError("Model building timeout")

        except torch.cuda.CudaError:
            # Handle CUDA out of memory
            stats["GFLOPs"] = int(max_gflops * CUDA_CREATE_MULTIPLIER)
            logger.log(stats, step=0, epoch=0)
            raise RuntimeError("CUDA out of memory")

        
        try:
            stats["GFLOPs"] = get_gflops(model, cfg.training.batch_size, n_inputs,
                            image_size, device=device, verbose=verbose)
        except torch.cuda.CudaError:
            # Handle CUDA out of memory
            stats["GFLOPs"] = int(max_gflops * CUDA_CALCULATE_MULTIPLIER)
            logger.log(stats, step=0, epoch=0)
            raise RuntimeError("CUDA out of memory")


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


# Define a function to handle the timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Command timed out")

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