import copy
from typing import Tuple, List
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import sys

sys.path.append('../networks') # add parent directory

import numpy as np

from equivariant.nn import (
    FieldType,
    SequentialModule,
    GroupTensor,
)
from networks import (
    Restriction,
    EquivariantPool,
    EquivariantConv,
)
from equivariant.nn.modules import nonlinearities
from networks.eq_other import EquivariantNorm
from networks.CNNs.wrn import WideResNet
from networks.eq_resnet.util import (
    EquivariantWideConvBlock, 
    EquivariantWideConvBlock_vary_l, 
    EquivariantWideConvBlock_drop_out,
)

from networks.util import (
    calculate_output_image_size,
    get_fixed_params,
    get_gspace_from_name,
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
        num_channels: int = 3,
        layout: List[int] = [16, 16, 32, 64],
        num_classes: int = 10,
        kernel_layout: List[int] = [3,3],
        drop_out: float = 0.0,
        bias: bool = False,
        act_func: str = "ReLU",
        image_size: int = 32,
        average_adaptive_pooling: int = 1,
        **kwargs,
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
                num_channels=num_channels,
                layout=layout,
                kernel_layout=self.kernel_layout,
                drop_out=self.drop_out,
                bias=self.bias,
            )
        
        self.set_name(depth=depth, k=k, kernel_layout=kernel_layout, group=group, rotation=rotation)

        gspace = get_gspace_from_name(group, rotation)
        self.gspace = gspace

        self.num_channels = np.array(layout, dtype=float)
        # Add width
        self.num_channels = (self.num_channels * np.array([1, k, k, k]))
        self.num_channels = np.round(self.num_channels).astype(int)

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * num_channels
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
        image_size = int(image_size[0] / 2)
        self.global_pool = nn.AdaptiveAvgPool2d(average_adaptive_pooling) 
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size * average_adaptive_pooling * average_adaptive_pooling, num_classes
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
            pass
            #equi_param = get_param_count(self)
            #print(f"Equivariant_WRN params: {equi_param}")
            

    def _wide_layer(
        self,
        block,
        out_channels: int,
        num_blocks: int,
        stride: int,
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
        # x = F.avg_pool2d(x, (2,2)) if x.shape[-1] > 1 else x
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
        

    def set_name(self, depth, k, kernel_layout, group, rotation):
        value = ""
        for i, element in enumerate(kernel_layout):
            value += str(element) 
            if i != len(kernel_layout) - 1:
                value += ","

        self.name = f"eq_wrn_{depth}_{k:.1f}_K{value}_" 
        if group == "cyclic":
            group_id = "C"
        elif group == "dihedral":
            group_id = "D"
        else:
            ValueError("group not recognized")

        self.name += f"{group_id}{rotation}"
        print(self.name)