from datetime import time
import math
import timeit
import warnings
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import hydra
from omegaconf import DictConfig
import sys
import copy
sys.path.append('../scaling-laws-ecnn') # add parent directory
import os
#os.environ['HYDRA_FULL_ERROR'] = '1'

import numpy as np

from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    SequentialModule,
    GroupTensor,
)
from networks import (
    Restriction,
    EquivariantPool,
    EquivariantConvBlock,
    EquivariantConv,
)
from nn.modules import nonlinearities
from networks.eq_layers import EquivariantNorm
from networks.wrn import WideResNet
from networks.eq_wrn_util import (
    EquivariantWideConvBlock, 
    EquivariantWideConvBlock_vary_l, 
    EquivariantWideConvBlock_drop_out,
)

from networks.util import (
    calculate_output_image_size,
    get_gspace,
)

CHANNELS_CONSTANT = 1

class EquivariantWideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 4,
        group: str = "cyclic",
        rotation: int = 4,
        fix_params: bool = False,
        restrict: List[str] = [None, None],  # "invariant", "reflection", "halved"
        input_channels: int = 3,
        layout: List[int] = [16, 16, 32, 64],
        kernel_size: int = 3,
        padding: int = 1,
        num_classes: int = 10,
        kernel_layout: List[int] = [3,3],
        drop_out: float = 0.0,
        bias: bool = False,
        act_func: str = "ReLU",
        image_size: int = 32,
    ):
        self.depth = depth
        self.widen_factor = widen_factor
        self.group = group
        self.rotation = rotation
        self.fix_params = fix_params
        self.restrict = [None, restrict] if isinstance(restrict, str) or restrict is None else restrict
        self.restrict = list(self.restrict)
        assert len(self.restrict) == 2, "restrict must be a string or a list of two strings"
        self.input_channels = input_channels
        self.layout = layout
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_classes = num_classes
        self.kernel_layout = kernel_layout
        self.drop_out = drop_out
        self.bias = bias
        self.act_func = act_func

        super(EquivariantWideResNet, self).__init__()
        assert (self.depth - 4) % 6 == 0, "WideResNet depth should be 6n+4."
        n = (self.depth - 4) / 6
        if len(kernel_layout) == 1:
            n = int(n * 2)
        elif len(kernel_layout) == 4:
            n = int(n / 2)
        k = self.widen_factor

        if drop_out > 0.0:
            assert len(kernel_layout) == 2, "Dropout only implemented for kernel_layout = [3,3]"
            wide_conv_block = EquivariantWideConvBlock_drop_out
        elif len(kernel_layout) == 3 and not kernel_layout == [3,3,3] or len(kernel_layout) == 2:
            wide_conv_block = EquivariantWideConvBlock
        elif len(kernel_layout) in [1,3,4]:
            wide_conv_block = EquivariantWideConvBlock_vary_l
        else:
            raise ValueError("kernel_layout not recognized")

        if self.fix_params:
            self.wrn = WideResNet(
                depth=self.depth,
                num_classes=self.num_classes,
                widen_factor=self.widen_factor,
                input_channels=self.input_channels,
                layout=self.layout,
                kernel_size=self.kernel_size,
                padding=self.padding,
                kernel_layout=self.kernel_layout,
                drop_out=self.drop_out,
                bias=self.bias,
            )

        print("Eq_WRN_%d_%d" % (self.depth, k))
        gspace = get_gspace(group, rotation)


        self.num_channels = np.array(self.layout, dtype=float)
        # Add width
        self.num_channels = (self.num_channels * np.array([1, k, k, k])) / gspace.fibergroup.order()
        self.num_channels = np.round(self.num_channels).astype(int)

        # Heuristic to reduce number of parameters
        # heuristic is slower since binary search looks in the upper more expensive part of the channels
        # if self.fix_params:
        #     self.num_channels = calculate_fixed_params(self.num_channels,
        #                                                self.gspace, self.restrict)

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.input_channels
        )

        # "Lifting" conv from trivial to regular feature fields
        self.conv1 = EquivariantConv(
            in_type=self.input_field_type,
            out_channels=int(self.num_channels[0]),
            kernel_size=self.kernel_size,
            padding=int(self.padding),
            groups=1,
            stride=1,
            dilation=1,
            bias=bias,
        )
        if self.fix_params:
            self.conv1 = self.iter_fix_param(0, self.conv1, self.wrn.conv1)
            preserved_field_type = self.conv1.out_type
        image_size = calculate_output_image_size(image_size, stride=1)
            

        self.field_type = self.conv1.out_type
        self.layer1 = self._wide_layer(
            wide_conv_block,
            self.num_channels[1],
            n,
            stride=1,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            act_func=self.act_func,
            kernel_layout=self.kernel_layout,
        )
        if self.fix_params:
            self.layer1 = self.iter_fix_param(1, self.layer1, self.wrn.layer1, wide_conv_block, 
                                              preserved_field_type, n=n, stride=1)
        preserved_field_type = self.layer1.out_type
        image_size = calculate_output_image_size(image_size, stride=1)
        
        self.restrict1 = Restriction(preserved_field_type, self.group, self.rotation, self.restrict[0])
        self.field_type = self.restrict1.out_type
        preserved_field_type = self.field_type

        self.layer2 = self._wide_layer(
            wide_conv_block,
            self.num_channels[2],
            n,
            stride=2,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            act_func=self.act_func,
            kernel_layout=self.kernel_layout,
        )
        if self.fix_params:
            self.field_type = preserved_field_type
            self.layer2 = self.iter_fix_param(2, self.layer2, self.wrn.layer2, wide_conv_block, 
                                              preserved_field_type, n=n, stride=2)
        image_size = calculate_output_image_size(image_size, stride=2)

        # Restrict last conv and res layers
        self.restrict2 = Restriction(self.layer2.out_type, self.group, self.rotation, self.restrict[1])
        self.field_type = self.restrict2.out_type

        if self.fix_params:
            preserved_field_type = self.field_type

        self.layer3 = self._wide_layer(
            block=wide_conv_block,
            out_channels=self.num_channels[3],
            num_blocks=n,
            stride=2,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            act_func=self.act_func,
            kernel_layout=self.kernel_layout,
        )
        if self.fix_params:
            self.field_type = preserved_field_type
            self.layer3 = self.iter_fix_param(3, self.layer3, self.wrn.layer3, 
                                              wide_conv_block, preserved_field_type, n=n, stride=2)
        image_size = calculate_output_image_size(image_size, stride=2)

        self.bn1 = EquivariantNorm(self.layer3.out_type, affine=False)
        self.relu = getattr(nonlinearities, act_func)(self.bn1.out_type)

        self.invariant_map = EquivariantPool(self.relu.out_type, invariant_map=True)
        image_size = int(image_size[0] / 2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size * image_size * image_size, self.num_classes
        )

        if self.fix_params:
            # size of wrn total and size of equivariant part
            norm_para = sum([p.numel() for p in self.wrn.parameters() if p.requires_grad])
            del self.wrn
            equi_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
            current_ratio = equi_param / norm_para
            print(f"Equivariant_WRN / WRN parameter ratio: {current_ratio:.3f}")
            

    def _wide_layer(
        self,
        block,
        out_channels: int,
        num_blocks: int,
        stride: int,
        kernel_size: int,
        padding: int,
        bias: bool,
        kernel_layout: List[int],
        act_func: str,
    ):
        # num_blocks is n in wide resnet paper
        # how many layers each block has
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    self.field_type,
                    out_channels,
                    stride=stride,
                    kernel_size=kernel_size,
                    padding=padding,
                    kernel_layout=kernel_layout,
                    bias=bias,
                    act_func=act_func,
                )
            )
            self.field_type = layers[-1].out_type

        return SequentialModule(*layers)

    def forward(self, x):
        # Wrap input tensor in a GroupTensor
        x = GroupTensor(x, self.input_field_type)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.restrict1(x)
        x = self.layer2(x)
        x = self.restrict2(x)
        x = self.layer3(x)
        x = self.relu(self.bn1(x))
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = F.avg_pool2d(x, 2) if x.shape[-1] > 1 else x
        x = self.flatten(x)
        x = self.classifier(x)
        return x
        
    
    def iter_fix_param(self, l, equi_conv_block, normal_conv_block, block=None,
                       preserved_field_type=None, n=None, stride=None):
        norm_param = sum([p.numel() for p in normal_conv_block.parameters() if p.requires_grad])
        equi_param = sum([p.numel() for p in equi_conv_block.parameters() if p.requires_grad])
        current_channel_size = self.num_channels[l]
        old_equi_param = None

        # initialize search range
        if equi_param > norm_param:
            lower_bound = max(current_channel_size - 200, 1)
            upper_bound = current_channel_size
        else:
            lower_bound = current_channel_size
            upper_bound = int(self.layout[l] // 0.7) 
            if l == 1:
                upper_bound = max(upper_bound, int(self.layout[l]) + 20)

        # binary search
        while lower_bound <= upper_bound:
            prediction = (lower_bound + upper_bound) // 2
            # save the old one since we might not be in 1% range
            old_equi_param, old_equi_conv_block = equi_param, equi_conv_block 
            equi_param, equi_conv_block = self.param_count(l,
                                        prediction, block, preserved_field_type, n, stride)

            if abs(equi_param - norm_param) < 0.01:
                last_ratio = equi_param / norm_param
                return equi_conv_block

            if equi_param < norm_param:
                # prediction is too small
                lower_bound = prediction + 1
            else:
                upper_bound = prediction - 1

        # if no solution found, return closest channel size
        if old_equi_param is not None:
            if abs(old_equi_param - norm_param) < abs(equi_param - norm_param):
                equi_conv_block = old_equi_conv_block
            
        last_ratio = equi_param / norm_param
        print(f'Ratio for block {l}: {last_ratio}')
        return equi_conv_block


    def param_count(self, l, channel_size_prediction, block, preserved_field_type=None, n=None, stride=None):
        if l == 0:
            # change the conv1
            eq_conv_block = EquivariantConv(
                in_type=self.input_field_type,
                out_channels=channel_size_prediction,
                kernel_size=self.kernel_size,
                padding=self.padding,
                groups=1,
                stride=1,
                dilation=1,
                bias=self.bias,
            )
        else:
            self.field_type = copy.deepcopy(preserved_field_type)
            eq_conv_block = self._wide_layer(
                block=block,
                out_channels=channel_size_prediction,
                num_blocks=n,
                stride=stride,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
                act_func=self.act_func,
                kernel_layout=self.kernel_layout,
            )
        return sum([p.numel() for p in eq_conv_block.parameters() if p.requires_grad]), eq_conv_block

@hydra.main(config_path="../experiment/conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # measure time
    start = timeit.default_timer()
    print(f"Kernel layout: ", cfg.model.kernel_layout)
    input_image_size = 410
    inp = torch.rand(1, 1, input_image_size, input_image_size)
    n_inputs = inp.shape[1]
    n_outputs = 10
    image_size=inp.shape[2]
    # depth, num_classes, widen_factor=1, dropRate=0.0
    #net = EquivariantWideResNet()
    net = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            image_size=image_size,
        )
    # tot_param = sum([p.numel() for p in net.conv1.parameters()  if p.requires_grad])
    tot_param = sum([p.numel() for p in net.parameters()  if p.requires_grad])
    print(f'Total number of parameters: {tot_param}') # total 2.748.890 # block1 121248
    #print(net.layer1)

    inp = inp.cuda()
    net.cuda()
    print(net(inp).size())

    # measure time
    stop = timeit.default_timer()
    print(f"Time elapsed: {stop - start}")

    #y = net(torch.randn(1,3,32,32))
    #print(y.size())


if __name__ == "__main__":
    main()