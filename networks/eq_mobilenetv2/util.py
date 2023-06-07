from typing import Tuple, List
from torch import nn
import numpy as np
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

from nn import (
    GroupTensor,
    FieldType,
    EquivariantModule,
    SequentialModule,

)
from networks.eq_convs import Equivariant_Conv_BN_actF, EquivariantConv


class EquivariantBottleneck(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        act_func: str = "Mish", # Mish or ReLU
        expand_ratio: int = 6,
    ):
        super(EquivariantBottleneck, self).__init__()
        self.in_type = in_type

        # we only have a residual connection 
        # if stride is 1 and the channel number does not change 
        self.residual_connection = (stride == 1 and len(in_type) == out_channels)
        expanded_num_channels = int(np.round(len(self.in_type) * expand_ratio))

        # 1. Block with 1x1 equivariant convolution
        self.conv1 = Equivariant_Conv_BN_actF(
            in_type=self.in_type,
            out_channels=expanded_num_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            dilation=dilation,
            bias=bias,
            act_func=act_func,
        )
        
        # 2. Block with 3x3 equivariant convolution
        self.conv2 = Equivariant_Conv_BN_actF(
            in_type=self.conv1.out_type,
            out_channels=expanded_num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
            act_func=act_func,
            groups=expanded_num_channels,
        )

        # 3. Block with 1x1 equivariant convolution
        self.conv3 = Equivariant_Conv_BN_actF(
            in_type=self.conv2.out_type,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            dilation=dilation,
            bias=bias,
            act_func="None",
        )
        self.out_type = self.conv3.out_type

        self.shortcut = nn.Identity()
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = EquivariantConv(
                self.in_type,
                len(self.conv3.out_type),
                kernel_size=1,
                padding=0,
                stride=stride,
                bias=False,
            )

    def forward(self, input: GroupTensor) -> GroupTensor:
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.shortcut(input) + x 
        return x 

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
        

class EquivariantBottleneckBlock(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        act_func: str = "Mish", # Mish or ReLU
        expand_ratio: int = 6,
        num_blocks: int = 1,
    ):
        #print('Block')
        super(EquivariantBottleneckBlock, self).__init__()
        self.in_type = in_type
        self.layers = \
            [
                EquivariantBottleneck(
                    in_type=self.in_type,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    act_func=act_func,
                    expand_ratio=expand_ratio,
                )
            ]
        self.layers.extend\
            ([
                EquivariantBottleneck(
                    in_type=self.layers[-1].out_type,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=1,
                    dilation=dilation,
                    bias=bias,
                    act_func=act_func,
                    expand_ratio=expand_ratio,
                )
                for _ in range(num_blocks - 1)
            ])
        self.out_type = self.layers[-1].out_type
        self.block = SequentialModule(*self.layers)
        #print('number of blocks: ', len(self.layers))

    def forward(self, input: GroupTensor) -> GroupTensor:
        x = self.block(input)
        return x
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

