import timeit
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import sys

sys.path.append('../scaling-laws-ecnn') # add parent directory

from networks import (
    e2wrn28_7R,
)
from networks.util import (
    get_param_count
)


@hydra.main(config_path="../experiment/conf", config_name="train_e2", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # measure time
    start = timeit.default_timer()
    input_image_size = 32
    inp = torch.rand(1, 3, input_image_size, input_image_size)
    n_inputs = inp.shape[1]
    n_outputs = 10
    image_size=inp.shape[2]
    #net = e2wrn28_7R()
    # depth, num_classes, widen_factor=1, dropRate=0.0
    #net = EquivariantWideResNet()
    net = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            # image_size=image_size,
        )
    print(f'Total number of parameters: {get_param_count(net)}') # total 2.748.890 # block1 121248
    #print(net.layer1)

    inp = inp.cuda()
    net.cuda()
    print(net(inp).size())

    # measure time
    stop = timeit.default_timer()
    print(f"Time elapsed: {stop - start}")


if __name__ == "__main__":
    main()