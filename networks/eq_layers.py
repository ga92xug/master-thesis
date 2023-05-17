from typing import Tuple, List
from torch import nn
import numpy as np
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory
import nn as nn_eq

from nn import (
    GroupTensor,
    FieldType,
    EquivariantModule,
    SequentialModule,
    R2Conv,
    GroupNorm,
    InducedNormGroupNorm,
    GroupStandardization,
    BatchNorm,
    InducedNormBatchNorm,
    Mish,
    ReLU,
    NormNonLinearity,
    InducedGatedNonLinearity,
    GroupPooling,
    NormPool,
    InducedNormPool,
    NormAvgPool,
    NormMaxPool,
    PointwiseAvgPool,
    PointwiseAdaptiveAvgPool,
    PointwiseMaxPool,
    DisentangleModule,
    RestrictionModule,
    MultipleModule,
)
from group_theory import Representation
from nn.modules import nonlinearities

__all__ = [
    "Restriction",
    "EquivariantConv",
    "EquivariantNorm",
    "EquivariantPool",
    "EquivariantConvBlock",
    "wide_conv_block",
    # MobileNetV2
    "EquivariantConvBlock_Conv_BN_actF",
    "EquivariantBottleneck",
    "EquivariantBottleneckBlock",
]


def only_zero_freq(repr: Representation):
    for irr in repr.irreps:
        idx = list(
            filter(
                lambda x: repr.group.irreps()[x].name == f"irrep_{irr[0]},{irr[1]}",
                range(len(repr.group.irreps())),
            )
        )
        if repr.group.irreps()[idx[0]].attributes["frequency"] != 0:
            return False
    return True


class Restriction(EquivariantModule):
    def __init__(
        self, in_type: FieldType, group: str, rotation: int, restrict: str = None
    ):
        super().__init__()
        self.in_type = in_type
        if restrict == "none":
            self.restrict = nn.Identity()
            self.out_type = self.in_type
        else:
            layers = list()

            if restrict == "reflection":
                assert group != "cyclic", "Cyclic groups can't be restricted to reflection."
                subgroup_id = (np.pi, 1) if group == "orthogonal" else (0, 1)

            elif restrict == "halved":
                assert group != "orthogonal", "Orthogonal group can't be restricted by halve."
                assert rotation % 2 == 0, f"Number of rotations ({rotation}) is not divisible by 2."
                subgroup_id = (0, rotation // 2) if group == "dihedral" else (rotation // 2)
                
            elif restrict == "invariant":  
                # restrict to invariant case
                subgroup_id = (None, 1) if group != "cyclic" else 1
            else:
                raise ValueError(f"Restriction {restrict} not implemented.")

            layers.append(RestrictionModule(self.in_type, subgroup_id))
            layers.append(DisentangleModule(layers[-1].out_type))
            self.restrict = SequentialModule(*layers)
            self.out_type = self.restrict.out_type

    def forward(self, x):
        return self.restrict(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


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


class EquivariantNorm(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        affine: bool = False,
    ):
        super().__init__()
        self.in_type = in_type

        # batch norm
        norm = BatchNorm
        induced_norm = InducedNormBatchNorm
        param = affine

        # Split into pointwise and induced representations
        if len(set(self.in_type.fields_names())) == 1:
            self.norm = norm(self.in_type, param)
        else:
            labels = [
                "pointwise" if only_zero_freq(r) else "induced" for r in self.in_type
            ]
            field_type = self.in_type.group_by_labels(labels)
            modules = [
                (norm(field_type["pointwise"], param), "pointwise"),
                (induced_norm(field_type["induced"], param), "induced"),
            ]
            self.norm = MultipleModule(self.in_type, labels, modules)

        self.out_type = self.norm.out_type

    def forward(self, x):
        return self.norm(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantPool(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        pool_size: int = None,
        invariant_map: bool = False,
    ):
        super().__init__()
        self.in_type = in_type
        self.map = None
        self.pool = None

        # Split into pointwise and induced representations
        if len(set(self.in_type.fields_names())) == 1:
            labels = ["pointwise"] * len(self.in_type)
            modules_map = [(GroupPooling(self.in_type), "pointwise")]
            modules_pool = [(PointwiseMaxPool(self.in_type, pool_size), "pointwise")]
        # Induced
        else:
            labels = [
                "pointwise" if only_zero_freq(r) else "induced" for r in self.in_type
            ]
            out_type = self.in_type.group_by_labels(labels)
            modules_map = [
                (NormPool(out_type["pointwise"]), "pointwise"),
                (InducedNormPool(out_type["induced"]), "induced"),
            ]
            modules_pool = [
                (PointwiseMaxPool(out_type["pointwise"], pool_size), "pointwise"),
                (NormMaxPool(out_type["induced"], pool_size), "induced"),
            ]

        if invariant_map:
            self.map = MultipleModule(self.in_type, labels, modules_map)
            if pool_size is not None:
                self.pool = PointwiseMaxPool(self.out_type, pool_size)
                self.out_type = self.pool.out_type
            else:
                self.pool = nn.Identity()
                self.out_type = self.map.out_type
        elif pool_size is not None:
            self.pool = MultipleModule(self.in_type, labels, modules_pool)
            self.out_type = self.pool.out_type
        else:
            raise ValueError(
                "EquivariantPool got no values for pooling and/or invariant mapping."
            )

    def forward(self, x):
        if self.map:
            x = self.map(x)
        x = self.pool(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantConvBlock(EquivariantModule):
    # groups is for induced representations
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
        self.conv1 = EquivariantConvBlock_Conv_BN_actF(
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
        self.conv2 = EquivariantConvBlock_Conv_BN_actF(
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
        self.conv3 = EquivariantConvBlock_Conv_BN_actF(
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


class EquivariantConvBlock_Conv_BN_actF(EquivariantModule):
    # groups is for induced representations
    # num_groups is for equivariant normalization
    # default is None so BatchNorm is used
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

        # Induced
        if self.in_type.gspace.fibergroup.name == "O(2)":
            NotImplementedError("Induced ConvBlock not implemented")
        # Cyclic and Dihedral Groups
        else:
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

if __name__ == "__main__":
    # 
    input_channels = 3
    out_channels = 3
    r2_act = nn_eq.rot2dOnR2(N=4)
    r2_in_type = FieldType(r2_act, [r2_act.trivial_repr]*input_channels)
    
    eq_conv = EquivariantConv(in_type=r2_in_type, out_channels=out_channels, \
                              kernel_size=3, padding=1, stride=1, bias=False)
    

    conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)

    tot_param = sum([p.numel() for p in eq_conv.parameters() if p.requires_grad])
    print('Total number of parameters: {}'.format(tot_param))
    tot_param = sum([p.numel() for p in conv1.parameters() if p.requires_grad])
    print('Total number of parameters: {}'.format(tot_param))