from omegaconf import DictConfig
import torch
import numpy as np
import hydra

from group_theory import kernels
import nn


np.set_printoptions(precision=3, linewidth=10000, suppress=True)

from networks import *

@hydra.main(config_path="../experiment/conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(f"Kernel layout: ", cfg.model.kernel_layout)
    inp = torch.rand(1, 1, 32, 32)
    n_inputs = inp.shape[1]
    n_outputs = 10
    # depth, num_classes, widen_factor=1, dropRate=0.0
    #net = EquivariantWideResNet()
    net = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
        )
    # tot_param = sum([p.numel() for p in net.conv1.parameters()  if p.requires_grad])
    tot_param = sum([p.numel() for p in net.parameters()  if p.requires_grad])
    print('Total number of parameters: {}'.format(tot_param)) # total 2.748.890 # block1 121248
    #print(net.layer1)

    inp = inp# .cuda()
    net #.cuda()
    print(net(inp).size())

    #y = net(torch.randn(1,3,32,32))
    #print(y.size())


if __name__ == "__main__":
    main()

