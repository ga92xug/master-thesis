import signal
from typing import List, Tuple
from lightning import Callback
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
        image_size: int,
    ) -> Tuple[float, float]:
    net.train()
    input_tensor = torch.randn(batch_size, num_channels, \
            image_size, image_size).to(net.device)

    # FLOPS
    flops = FlopCountAnalysis(net, (input_tensor,))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    print(flop_count_table(flops, max_depth=5))
    gflops = flops.total() / 1e9
    gflops_per_image = gflops / batch_size
    print(f"GFLOPs per image: {gflops_per_image}")
    quit()
    
    # PARAMS
    param_count = parameter_count(net).get("net")
    #print(f"Number of parameters: {param_count}")
    mparam_count = param_count / 1e6
    
    return gflops_per_image, mparam_count

def timeout_handler(signum, frame):
    # Define a function to handle the timeout
    print("Model building time exceeded.")
    raise TimeoutError()


class ModelStats(Callback):
    def __init__(self, max_gflops: float, nas_trial: int):
        super().__init__()
        self.max_gflops = max_gflops
        if nas_trial < 0:
            self.is_nas = False
        else:
            self.is_nas = True

    def on_fit_start(self, trainer, pl_module):

        gflops_per_image, param_count = get_stats(
                net=pl_module, 
                batch_size=pl_module.hparams.dataset.batch_size, 
                num_channels=pl_module.hparams.num_channels, 
                image_size=pl_module.hparams.image_size,
            )
        
        hparams = {
            #"net_building_time": pl_module.net_building_time,
            "param_count": param_count,
            "GFLOPs_per_image": gflops_per_image,
        }

        for logger in trainer.loggers:
            logger.log_hyperparams(hparams)

        if self.is_nas and gflops_per_image > self.max_gflops:
            raise ValueError(f"GFLOPs {gflops_per_image} exceeds maximum allowed \
                             {self.max_gflops}")

        return
