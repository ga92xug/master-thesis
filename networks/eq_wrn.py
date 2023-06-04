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
    get_fixed_params,
    get_gspace,
    get_param_count,
    cuda_memory_usage,
)


class EquivariantWideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 4,
        group: str = "cyclic",
        rotation: int = 4,
        fix_params_mode: str = "no", # "iter", "heuristic", "all"
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
        restrict = [None, restrict] if isinstance(restrict, str) else restrict
        restrict = list(restrict)
        assert len(restrict) == 2, "restrict must be a string or a list of two strings"
        # self.padding = padding
        self.kernel_layout = kernel_layout
        if len(kernel_layout) != 2 or kernel_layout[0] == 3 or kernel_layout[1] == 3:
            self.padding = 1
        elif kernel_layout == [5,5]:
            self.padding = 2
        elif kernel_layout == [7,7]:
            self.padding = 3
        if rotation > 4 and self.kernel_layout[0] == 3:
            pass
            # warnings.warn(f"Discretization artifacts are expected for rotation > 4 and \
            #         kernel_layout = {self.kernel_layout}.")
        self.drop_out = drop_out
        self.bias = bias
        self.act_func = act_func

        super(EquivariantWideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n+4."
        self.fix_params_mode = fix_params_mode
        n = (depth - 4) / 6
        if len(kernel_layout) == 1:
            n = int(n * 2)
        elif len(kernel_layout) == 4:
            n = int(n / 2)
        k = widen_factor

        if drop_out > 0.0:
            assert len(kernel_layout) == 2, "Dropout only implemented for kernel_layout = [3,3]"
            wide_conv_block = EquivariantWideConvBlock_drop_out
        elif len(kernel_layout) == 3 and not kernel_layout == [3,3,3] or len(kernel_layout) == 2:
            wide_conv_block = EquivariantWideConvBlock
        elif len(kernel_layout) in [1,3,4]:
            wide_conv_block = EquivariantWideConvBlock_vary_l
        else:
            raise ValueError("kernel_layout not recognized")

        if self.fix_params_mode == "iter":
            self.wrn = WideResNet(
                depth=depth,
                num_classes=num_classes,
                widen_factor=widen_factor,
                input_channels=input_channels,
                layout=layout,
                kernel_size=self.kernel_size,
                padding=self.padding,
                kernel_layout=self.kernel_layout,
                drop_out=self.drop_out,
                bias=self.bias,
            )
        value = ""
        for i, element in enumerate(kernel_layout):
            value += str(element) 
            if i != len(kernel_layout) - 1:
                value += ","

        print(f"Eq_WRN_{depth}_{k:.2f}_B({value})")
        gspace = get_gspace(group, rotation)
        self.gspace = gspace

        self.num_channels = np.array(layout, dtype=float)
        # Add width
        self.num_channels = (self.num_channels * np.array([1, k, k, k]))
        self.num_channels = np.round(self.num_channels).astype(int)

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * input_channels
        )

        # "Lifting" conv from trivial to regular feature fields
        kwargs = {"in_type": self.input_field_type, "out_channels": self.num_channels[0],
            'kernel_size': 5, "padding": 2, "groups": 1, 'bias': bias}
        normal_conv = self.wrn.conv1 if fix_params_mode in ["all", "iter"] else None
        self.conv1 = get_fixed_params(EquivariantConv, fix_params_mode, 
                        normal_block=normal_conv, gspace=self.input_field_type.gspace, 
                        **kwargs)
        image_size = calculate_output_image_size(image_size, stride=1) # 32
        
        self.field_type = self.conv1.out_type
        normal_blocks = self.wrn.layer1 if fix_params_mode in ["all", "iter"] else None
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
            normal_blocks=normal_blocks,
            drop_out=self.drop_out,
        )
        image_size = calculate_output_image_size(image_size, stride=1) # 32
        
        self.restrict1 = Restriction(self.layer1.out_type, group, rotation, restrict[0])
        self.field_type = self.restrict1.out_type

        normal_blocks = self.wrn.layer2 if fix_params_mode in ["all", "iter"] else None
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
            normal_blocks=normal_blocks,
            drop_out=self.drop_out,
        )
        image_size = calculate_output_image_size(image_size, stride=2) # 16

        # Restrict last conv and res layers
        #print(f"Restricting {self.layer2.out_type} to {self.restrict[1]}, {group}, {rotation}")
        self.restrict2 = Restriction(self.layer2.out_type, group, rotation, restrict[1])
        self.field_type = self.restrict2.out_type

        normal_blocks = self.wrn.layer3 if fix_params_mode in ["all", "iter"] else None
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
            normal_blocks=normal_blocks,
            drop_out=self.drop_out,
        )
        image_size = calculate_output_image_size(image_size, stride=2) # 8

        self.bn1 = EquivariantNorm(self.layer3.out_type, affine=False)
        self.relu = getattr(nonlinearities, act_func)(self.bn1.out_type)

        self.invariant_map = EquivariantPool(self.relu.out_type, invariant_map=True)
        image_size = int(image_size[0] / 8) 
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size * image_size * image_size, num_classes
        )

        # print stats
        if self.fix_params_mode in ["all", "iter"]:
            # size of wrn total and size of equivariant part
            norm_para = get_param_count(self.wrn)
            del self.wrn
            equi_param = get_param_count(self)
            current_ratio = equi_param / norm_para
            print(f"Equivariant_WRN / WRN parameter ratio: {current_ratio:.3f}")
        elif self.fix_params_mode in ["no", "heuristic"]:
            equi_param = get_param_count(self)
            print(f"Equivariant_WRN params: {equi_param}")
            

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
        normal_blocks = None,
        drop_out: float = 0.0,
    ):
        # num_blocks is n in wide resnet paper
        # how many layers each block has
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []


        for i, stride in enumerate(strides):
            if normal_blocks is not None:
                normal_block = normal_blocks.layer[i]
            else:
                normal_block = None

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
                    normal_block=normal_block,
                    fix_params_mode=self.fix_params_mode,
                    drop_out=drop_out,
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
        x = F.avg_pool2d(x, 8) if x.shape[-1] > 1 else x
        x = self.flatten(x)
        x = self.classifier(x)
        return x
        


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
