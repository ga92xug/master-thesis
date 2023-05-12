from typing import Tuple, List
from torch import nn
import numpy as np
import sys
from networks.util import cuda_memory_usage, get_fixed_params
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
    PointwiseDropout,
)
from group_theory import Representation
from nn.modules import nonlinearities
from networks.eq_layers import EquivariantNorm, EquivariantConv


# TODO check equivariantnorm

class EquivariantWideConvBlock(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        kernel_layout: List[int] = None,
        act_func: str = "ReLU", # ReLU
        normal_block = None, 
        fix_params_mode: str = "no",
        drop_out: float = 0.0,
    ):
        super(EquivariantWideConvBlock, self).__init__()
        assert drop_out == 0.0, "Dropout not implemented for this block"
        self.in_type = in_type
        self.kernel_layout = kernel_layout

        strides = np.ones_like(kernel_layout)
        paddings = np.zeros_like(kernel_layout)
        for i in range(len(kernel_layout)):
            if kernel_layout[i] == 3:
                strides[i] = stride
                break
        
        paddings = [padding if kernel_layout[i] > 1 else 0 for i in range(len(kernel_layout))]

        # block 1
        self.norm1 = EquivariantNorm(
            self.in_type, affine=False
        )
        self.act_func1 = getattr(nonlinearities, act_func)(self.norm1.out_type)
        kwargs = {"in_type": self.act_func1.out_type, "out_channels": out_channels,
            "kernel_size": kernel_layout[0], "padding": paddings[0], 
            "stride": strides[0], "dilation": dilation, "bias": bias}
        normal_conv = normal_block.conv1 if fix_params_mode in ["all", "iter"] else None        
        self.conv1 = get_fixed_params(EquivariantConv, fix_params_mode, normal_conv, 
                                      gspace=self.in_type.gspace, **kwargs)
        
        # self.conv1 = EquivariantConv(
        #     self.act_func1.out_type,
        #     out_channels,
        #     kernel_size=kernel_layout[0],
        #     padding=paddings[0],
        #     stride=strides[0],
        #     dilation=dilation,
        #     bias=bias,
        # )
        current_out_type = self.conv1.out_type
        
        if len(kernel_layout) == 3:
            norm = EquivariantNorm(self.conv1.out_type, affine=False)
            act = getattr(nonlinearities, act_func)(norm.out_type)
            kwargs = {"in_type": act.out_type, "out_channels": out_channels,
                "kernel_size": kernel_layout[1], "padding": paddings[1], 
                "stride": strides[1], "dilation": dilation, "bias": bias}
            normal_conv = normal_block.conv if fix_params_mode in ["all", "iter"] else None        
            conv = get_fixed_params(EquivariantConv, fix_params_mode, normal_conv, 
                                          gspace=act.out_type, **kwargs)
            # conv = EquivariantConv(
            #         act.out_type,
            #         out_channels,
            #         kernel_size=kernel_layout[1],
            #         padding=paddings[1],
            #         stride=strides[1],
            #         dilation=dilation,
            #         bias=bias,
            #     )
            current_out_type = conv.out_type
            self.block = SequentialModule(norm, act, conv)
        
        # block 2
        self.norm2 = EquivariantNorm(
            current_out_type, affine=False
        )
        self.act_func2 = getattr(nonlinearities, act_func)(self.norm2.out_type)
        kwargs = {"in_type": self.act_func2.out_type, "out_channels": out_channels,
            "kernel_size": kernel_layout[-1], "padding": paddings[-1], 
            "stride": strides[-1], "dilation": dilation, "bias": bias}
        normal_conv = normal_block.conv2 if fix_params_mode in ["all", "iter"] else None
        self.conv2 = get_fixed_params(EquivariantConv, fix_params_mode, normal_conv,
                                        gspace=self.act_func2.out_type, **kwargs)
        # self.conv2 = EquivariantConv(
        #     self.act_func2.out_type,
        #     out_channels,
        #     kernel_size=kernel_layout[-1],
        #     padding=paddings[-1],
        #     stride=strides[-1],
        #     dilation=dilation,
        #     bias=bias,
        # )
        
        self.out_type = self.conv2.out_type

        self.shortcut = nn.Identity()
        if stride != 1 or self.in_type != self.out_type:
            norm = EquivariantNorm(
                self.in_type, affine=False
            )
            shortcut = EquivariantConv(
                norm.out_type,
                len(self.conv2.out_type),
                kernel_size=1,
                padding=0,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )
            
            self.shortcut = SequentialModule(*[norm, shortcut])

    def forward(self, x):
        # bn -> relu -> conv

        """
        x torch.Size([1, 64, 32, 32])
        out torch.Size([1, 256, 30, 30])
        Layer 1: 
        strides [1 1]
        paddings [0, 1]
        strides [1 1]
        paddings [0, 1]
        """
        #print("x", x.shape)
        out = self.norm1(x)
        out = self.act_func1(out)
        # print("out", out.shape)
        out = self.conv1(out)
        #print("out", out.shape)
        if len(self.kernel_layout) == 3:
            out = self.block(out)
            # print("out", out.shape)
        out = self.norm2(out)
        out = self.act_func2(out)   
        out = self.conv2(out)
        #print("out", out.shape)
        out += self.shortcut(x)
        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape



class EquivariantWideConvBlock_vary_l(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        kernel_layout: List[int] = None,
        act_func: str = "ReLU", # ReLU
        normal_block = None, 
        fix_params_mode: str = "no",
        drop_out: float = 0.0,
    ):
        super(EquivariantWideConvBlock_vary_l, self).__init__()
        assert drop_out == 0.0, "Dropout not implemented for this block"
        self.in_type = in_type
        self.kernel_layout = kernel_layout

        strides = np.ones_like(kernel_layout)
        for i in range(len(kernel_layout)):
            if kernel_layout[i] > 1:
                strides[i] = stride
                break
        
        paddings = [padding if kernel_layout[i] > 1 else 0 for i in range(len(kernel_layout))]
        #print("strides", strides)
        #print("paddings", paddings)

        # block 1
        self.norm1 = EquivariantNorm(
            self.in_type, affine=False
        )
        self.act_func1 = getattr(nonlinearities, act_func)(self.norm1.out_type)
        kwargs = {"in_type": self.act_func1.out_type, "out_channels": out_channels,
            "kernel_size": kernel_layout[0], "padding": paddings[0], 
            "stride": strides[0], "dilation": dilation, "bias": bias}
        normal_conv = normal_block.conv1 if fix_params_mode in ["all", "iter"] else None        
        self.conv1 = get_fixed_params(EquivariantConv, fix_params_mode, normal_conv, 
                                      gspace=self.in_type.gspace, **kwargs)
        conv = self.conv1
        current_out_type = self.conv1.out_type
        
        self.layer = []
        for i in range(1, len(kernel_layout)):
            norm = EquivariantNorm(current_out_type, affine=False)
            act = getattr(nonlinearities, act_func)(norm.out_type)
            kwargs = {"in_type": act.out_type, "out_channels": out_channels,
                "kernel_size": kernel_layout[i], "padding": paddings[i], 
                "stride": strides[i], "dilation": dilation, "bias": bias}
            normal_conv = normal_block.conv2 if fix_params_mode in ["all", "iter"] else None
            conv = get_fixed_params(EquivariantConv, fix_params_mode, normal_conv,
                                            gspace=act.out_type, **kwargs)
            current_out_type = conv.out_type
            self.layer.extend([norm, act, conv])
        
        self.layer = SequentialModule(*self.layer)
        self.out_type = current_out_type

        self.shortcut = nn.Identity()
        if stride != 1 or self.in_type != self.out_type:
            norm = EquivariantNorm(
                self.in_type, affine=False
            )
            shortcut = EquivariantConv(
                norm.out_type,
                len(conv.out_type),
                kernel_size=1,
                padding=0,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )
            
            self.shortcut = SequentialModule(*[norm, shortcut])

    def forward(self, x):
        # bn -> relu -> conv
        """
        x torch.Size([1, 64, 32, 32])
        out torch.Size([1, 256, 30, 30])
        Layer 1: 
        strides [1 1]
        paddings [0, 1]
        strides [1 1]
        paddings [0, 1]
        """
        #print("x", x.shape)
        out = self.norm1(x)
        out = self.act_func1(out)
        out = self.conv1(out)
        #print("out", out.shape)
        if len(self.kernel_layout) > 1:
            out = self.layer(out)
        #print("out", out.shape)
        out += self.shortcut(x)
        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantWideConvBlock_drop_out(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        kernel_layout: List[int] = None,
        act_func: str = "ReLU", # ReLU
        drop_out: float = 0.0,
        normal_block = None, 
        fix_params_mode: str = "no",
    ):
        super(EquivariantWideConvBlock_drop_out, self).__init__()
        self.in_type = in_type
        self.kernel_layout = kernel_layout

        strides = np.ones_like(kernel_layout)
        for i in range(len(kernel_layout)):
            if kernel_layout[i] > 1:
                strides[i] = stride
                break
        
        paddings = [padding if kernel_layout[i] > 1 else 0 for i in range(len(kernel_layout))]
        #print("strides", strides)
        #print("paddings", paddings)

        # block 1
        self.norm1 = EquivariantNorm(
            self.in_type, affine=False
        )
        self.act_func1 = getattr(nonlinearities, act_func)(self.norm1.out_type)
        kwargs = {"in_type": self.act_func1.out_type, "out_channels": out_channels,
            "kernel_size": kernel_layout[0], "padding": paddings[0], 
            "stride": strides[0], "dilation": dilation, "bias": bias}
        normal_conv = normal_block.conv1 if fix_params_mode in ["all", "iter"] else None        
        self.conv1 = get_fixed_params(EquivariantConv, fix_params_mode, normal_conv, 
                                      gspace=self.in_type.gspace, **kwargs)
        
        # self.conv1 = EquivariantConv(
        #     self.act_func1.out_type,
        #     out_channels,
        #     kernel_size=kernel_layout[0],
        #     padding=paddings[0],
        #     stride=strides[0],
        #     dilation=dilation,
        #     bias=bias,
        # )
                
        self.norm2 = EquivariantNorm(self.conv1.out_type, affine=False)
        self.act_func2 = getattr(nonlinearities, act_func)(self.norm2.out_type)
        self.drop_out = PointwiseDropout(in_type=self.act_func2.out_type, p=drop_out)
        kwargs = {"in_type": self.drop_out.out_type, "out_channels": out_channels,
            "kernel_size": kernel_layout[-1], "padding": paddings[-1], 
            "stride": strides[-1], "dilation": dilation, "bias": bias}
        normal_conv = normal_block.conv2 if fix_params_mode in ["all", "iter"] else None
        self.conv2 = get_fixed_params(EquivariantConv, fix_params_mode, normal_conv,
                                        gspace=self.act_func2.out_type, **kwargs)
        
        self.out_type = self.conv2.out_type

        self.shortcut = nn.Identity()
        if stride != 1 or self.in_type != self.out_type:
            norm = EquivariantNorm(
                self.in_type, affine=False
            )
            shortcut = EquivariantConv(
                norm.out_type,
                len(self.conv2.out_type),
                kernel_size=1,
                padding=0,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )
            
            self.shortcut = SequentialModule(*[norm, shortcut])

    def forward(self, x):
        # bn -> relu -> conv
        # print("\nBlock")
        # print("norm1")
        # cuda_memory_usage()
        out = self.norm1(x)
        #cuda_memory_usage()
        #print("act_func1")
        out = self.act_func1(out)
        #cuda_memory_usage()
        #print("conv1")
        out = self.conv1(out)
        
        
        #print("norm2")
        #cuda_memory_usage()
        out = self.norm2(out)
        #cuda_memory_usage()
        #print("act_func2")
        out = self.act_func2(out)
        #cuda_memory_usage()
        out = self.drop_out(out)
        #cuda_memory_usage()
        out = self.conv2(out)
        #cuda_memory_usage()
        out += self.shortcut(x)
        #cuda_memory_usage()
        #print("\n")
        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape