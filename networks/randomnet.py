import warnings
import numpy as np
from typing import Tuple
import torch
from torch import nn
import sys

from networks.eq_layers import EquivariantConvBlock
sys.path.append('../scaling-laws-ecnn') # add parent directory


class RandomNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        return torch.rand(x.shape[0], self.num_classes).cuda()
        return x


if __name__ == "__main__":
    inp = torch.rand(1, 3, 32, 32).cuda()
    model = RandomNet(10).cuda()
    out = model(inp)
    print(out.shape)
