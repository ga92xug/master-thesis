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
)
from nn.modules import nonlinearities


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
        in_channels: int,
        squeeze_channels: int,
    ):
        super(EquivariantSqueezeExcitation, self).__init__()
        self.in_type = in_type

        self.fc1 = EquivariantConv(
            self.in_type, squeeze_channels, kernel_size=1, padding=0
        )

        self.fc2 = EquivariantConv(
            self.fc1.out_type, in_channels, kernel_size=1, padding=0
        )

        self.out_type = self.fc2.out_type

    def forward(self, input: GroupTensor):
        x = self.fc1(input)
        x = self.fc2(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    

    
class Eq_Conv2dSamePadding(EquivariantModule):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        image_size: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        # kernel_layout: List[int] = None,
    ):
        super().__init__()
        self.stride = [stride] * 2 if isinstance(stride, int) else stride
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.dilation = [dilation] * 2
        self.conv2d = EquivariantConv(in_type, out_channels, kernel_size, padding=0,
                                      stride=self.stride, groups=groups, bias=bias)
        self.out_type = self.conv2d.out_type
        

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        # kh, kw = self.weight.size()[-2:]
        kh, kw = kernel_size, kernel_size # we don't support uneven kernel sizes
        sh, sw = self.stride[0], self.stride[1]
        # types of ih, sh, iw and sw
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        self.pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        self.pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        
    def forward(self, x):
        if self.pad_h > 0 or self.pad_w > 0:
            x.tensor = nn.functional.pad(x.tensor, 
                            pad=(self.pad_w // 2, self.pad_w - self.pad_w // 2, 
                            self.pad_h // 2, self.pad_h - self.pad_h // 2))
        x = self.conv2d(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
