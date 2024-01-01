import signal
from typing import List, Tuple
import wandb
import hydra
from hydra import compose, initialize
import timeit
import torch
from omegaconf import DictConfig
from fvcore.nn import (
    FlopCountAnalysis, 
    flop_count_table, 
    parameter_count_table,
    parameter_count
)

def get_stats(
        net: torch.nn.Module, 
        batch_size: int, 
        num_channels: int, 
        image_size: int
    ) -> Tuple[float, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # FLOPS
    input_tensor = torch.randn(batch_size, num_channels, \
            image_size, image_size).to(device)
    flops = FlopCountAnalysis(net, (input_tensor,))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    gflops = flops.total() / 1e9
    gflops_per_image = gflops / batch_size
    
    # PARAMS
    param_count = parameter_count(net)
    mparam_count = param_count / 1e6
    
    return gflops_per_image, mparam_count

def timeout_handler(signum, frame):
    # Define a function to handle the timeout
    print("Model building time exceeded.")
    raise TimeoutError()