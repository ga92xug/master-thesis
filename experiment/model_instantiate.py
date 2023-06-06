import hydra
import timeit
import torch
from omegaconf import DictConfig
from fvcore.nn import FlopCountAnalysis, flop_count_table
from networks.util import get_gflops, get_param_count


def create_model(
        cfg: DictConfig, 
        n_inputs: int, 
        n_outputs: int, 
        image_size: int,
        device: torch.device,
        verbose: int = 1,
):
    """
    Instantiate the model and return:
    - number of parameters
    - model building time
    - train time
    - GFLOPs
    """

    stats = {}

    start = timeit.default_timer()
    model = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            image_size=image_size,
        ).to(device)
    stop = timeit.default_timer()
    model_building_time = stop - start
    stats["model_building_time"] = model_building_time
    stats["param_count"] = get_param_count(model, in_mb=False, verbose=verbose)

    stats["GFLOPs"] = get_gflops(model, cfg.training.batch_size, n_inputs, 
                        image_size, verbose=verbose)

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