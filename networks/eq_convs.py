import math
from typing import Tuple, List
from torch import nn
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

from networks.eq_other import EquivariantNorm, EquivariantPool

from nn import (
    GroupTensor,
    FieldType,
    EquivariantModule,
    R2Conv,
    NormNonLinearity,
    Swish,
    PointwiseAdaptiveAvgPool,
)
from nn.modules import nonlinearities

PADDINGS = {
    1: 0,
    3: 1,
    5: 2,
    7: 3,
}

class EquivariantConv(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        self.in_type = in_type  # declaration required by base class

        # Cyclic and Dihedral Groups
        out_type = FieldType(
            self.in_type.gspace,
            [self.in_type.gspace.regular_repr] * out_channels,
        )
        self.conv = R2Conv(
            self.in_type,
            out_type,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            sigma=None,
            frequencies_cutoff=lambda r: 3 * r,
        )
        self.trivials, self.gate = None, None

        self.out_type = self.conv.out_type

    def forward(self, x):
        return self.conv(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

class EquivariantConvChangeFactor(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        change_factor: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        self.in_type = in_type  # declaration required by base class

        out_channels = int(round(change_factor * len(in_type)))
        if out_channels < 1:
            out_channels = 1

        # Cyclic and Dihedral Groups
        out_type = FieldType(
            self.in_type.gspace,
            [self.in_type.gspace.regular_repr] * out_channels,
        )
        self.conv = R2Conv(
            self.in_type,
            out_type,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            sigma=None,
            frequencies_cutoff=lambda r: 3 * r,
        )
        self.trivials, self.gate = None, None

        self.out_type = self.conv.out_type
        #assert int(round(change_factor * len(in_type))) == len(self.out_type), f"{len(in_type)} * {change_factor} != {len(self.out_type)}"

    def forward(self, x):
        return self.conv(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

class EquivariantConvBlock(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        pool_size: int = None,
        invariant_map: bool = False,
        act_func: str = "ReLU",
    ):
        super().__init__()
        self.in_type = in_type  # declaration required by base class
        self.conv = EquivariantConv(
            self.in_type,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        self.act_func = getattr(nonlinearities, act_func)(self.conv.out_type)

        self.norm = EquivariantNorm(
            self.act_func.out_type, affine=False
        )

        if pool_size is not None or invariant_map:
            self.pool = EquivariantPool(self.norm.out_type, pool_size, invariant_map)
            self.out_type = self.pool.out_type
        else:
            self.pool = nn.Identity()
            self.out_type = self.norm.out_type

    def forward(self, x):
        x = self.conv(x)
        x = self.act_func(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class Equivariant_Conv_BN_actF(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        act_func: str = "Mish", # "Mish" or "ReLU" or "None"
    ):
        super().__init__()
        self.in_type = in_type  # declaration required by base class
        self.conv = EquivariantConv(
            in_type=self.in_type,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        self.norm = EquivariantNorm(
            self.conv.out_type, affine=True
        )        

        if act_func == "None":
            self.act_func = nn.Identity(self.conv.out_type)
        else:
            self.act_func = getattr(nonlinearities, act_func)(self.norm.out_type)

        self.out_type = self.norm.out_type

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act_func(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantSqueezeExcitation(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        sequeeze_ratio: float = 0.25,
    ):
        super(EquivariantSqueezeExcitation, self).__init__()
        self.in_type = in_type
        
        self.avgpool = PointwiseAdaptiveAvgPool(self.in_type, 1)

        self.conv1 = EquivariantConvChangeFactor(
            in_type=self.in_type, 
            change_factor=sequeeze_ratio, 
            kernel_size=1, 
            padding=0
        )

        self.act_func = Swish(self.conv1.out_type)

        self.conv2 = EquivariantConv(
            self.act_func.out_type, 
            len(self.in_type),
            kernel_size=1, 
            padding=0
        )

        self.scale_activation = NormNonLinearity(
            self.conv2.out_type, function="n_sigmoid"
        )

        self.out_type = self.scale_activation.out_type

    def _scale(self, input: GroupTensor):
        scale = self.avgpool(input)
        scale = self.conv1(scale)
        scale = self.act_func(scale)
        scale = self.conv2(scale)
        return self.scale_activation(scale)

    def forward(self, input: GroupTensor):
        scale = self._scale(input)
        return GroupTensor(scale.tensor * input.tensor, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    
class Eq_Conv2dSamePadding(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        # kernel_layout: List[int] = None,
    ):
        super().__init__()
        #print("out_channels", out_channels)
        padding = PADDINGS[kernel_size]
        self.conv2d = EquivariantConv(in_type, out_channels, kernel_size, 
                                      padding=padding, dilation=dilation,
                                      stride=stride, groups=groups, bias=bias)
        self.out_type = self.conv2d.out_type
        
    def forward(self, x):
        x = self.conv2d(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class Eq_Conv2dSamePaddingChangeFactor(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        change_factor: float,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        # kernel_layout: List[int] = None,
    ):
        super().__init__()
        #print("Eq_Conv2dSamePadding_constant_channel, in_type: ", in_type.size)
        self.in_type = in_type
        

        padding = PADDINGS[kernel_size]
        self.conv2d = EquivariantConvChangeFactor(
            in_type=in_type, 
            change_factor=change_factor, 
            kernel_size=kernel_size, 
            padding=padding,
            stride=stride, 
            dilation=dilation,
            bias=bias,
            groups=groups, 
        )
        self.out_type = self.conv2d.out_type

    def forward(self, x):
        x = self.conv2d(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


