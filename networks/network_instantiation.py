import timeit
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import sys

sys.path.append('../scaling-laws-ecnn') # add parent directory

# from networks import (
#     e2wrn28_7R,
# )
from networks.util import (
    get_param_count
)


@hydra.main(config_path="../experiment/conf", config_name="config", version_base="1.2")
def instantiate_model_forward_pass(cfg: DictConfig) -> None:    
    input_image_size = 224
    n_outputs = 10

    input_tensor = torch.rand(128, 3, input_image_size, input_image_size)
    n_inputs = input_tensor.shape[1]
    image_size=input_tensor.shape[2]

    start = timeit.default_timer()
    net = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            image_size=image_size,
        )
    stop = timeit.default_timer()
    model_building_time = stop - start
    print(f"Model building time: {model_building_time}")
    param_count = get_param_count(net)
    print(f'Total number of parameters: {param_count}')
    # print(net) 
    
    # time forward pass
    start = timeit.default_timer()
    net.train()
    for i in range(100):
        input_tensor = input_tensor.cuda()
        net.cuda()
        net(input_tensor)

    stop = timeit.default_timer()
    train_time = stop - start
    print(f"Train time elapsed: {train_time}")

    return param_count, model_building_time, train_time


if __name__ == "__main__":
    instantiate_model_forward_pass()
    #for rotation in range(2, 16, 2):
    #    param_count, model_building_time, train_time = instantiate_model_forward_pass()
    #    # f"model.rotation={rotation}"

    