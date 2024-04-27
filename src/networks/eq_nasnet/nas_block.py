from collections import OrderedDict
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import os
import sys
sys.path.append(f"{os.getcwd()}")

from src.networks.eq_nasnet.block_args import BlockArgs
from src.networks.equivariant_utils.eq_other import EquivariantPool
from src.networks.equivariant_utils.eq_convs import (
    EquivariantConv,
    EquivariantSqueezeExcitation,
    Eq_Conv2dSamePadding,
)
from src.networks.equivariant_utils.utils import (
    calculate_output_image_size, 
    adjusted_out_channels,
)
from equivariant.nn import (
    GroupTensor,
    FieldType,
    EquivariantModule,
    BatchNorm,
    Mish,
    ReLU,
    Swish,
    SequentialModule,
    PointwiseDropout,
)

class Eq_NAS_layer(EquivariantModule):
    """
    Block with variable content based on block_args.
    """
    def __init__(
            self, 
            in_type: FieldType, 
            in_channel_size, 
            fixed_params, 
            block_args: BlockArgs, 
            image_size, 
            dropout_rate=0.0,
            expand_ratio=2
        ):
        """
        Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
        """
        super().__init__()
        self.block_args = block_args
        self.original_in_type = in_type
        self.has_se = 0 < block_args.se_ratio < 1
        
        intermediate_channel_size = adjusted_out_channels(
            out_channel=in_channel_size,
            N=in_type.gspace.fibergroup.order(),
            fixed_params=fixed_params,
        )
        end_channel_size = adjusted_out_channels(
            out_channel=block_args.out_channel,
            N=in_type.gspace.fibergroup.order(),
            fixed_params=fixed_params,
        )

        # Expansion phase setup
        self.expand_ratio = expand_ratio if block_args.conv_op == 'mbconv' else 1
        intermediate_channel_size = int(round(intermediate_channel_size * self.expand_ratio))
        modules = OrderedDict()
        
        if self.expand_ratio != 1:
            modules['expand_bn'] = BatchNorm(in_type=in_type, affine=False)
            modules['expand_swish'] = Swish(in_type=modules['expand_bn'].out_type)
            modules['expand_conv'] = Eq_Conv2dSamePadding(
                in_type=modules['expand_swish'].out_type,
                out_channels=intermediate_channel_size,
                kernel_size=1,
                bias=False,
            )
            in_type = modules['expand_conv'].out_type  # Update in_type after expansion
        
        modules['bn1'] = BatchNorm(in_type=in_type, affine=False)
        modules['swish1'] = Swish(in_type=modules['bn1'].out_type)
        modules['conv1'] = Eq_Conv2dSamePadding(
            in_type=modules['swish1'].out_type,
            out_channels=intermediate_channel_size,
            kernel_size=block_args.kernel_size,
            groups=len(modules['swish1'].out_type) if block_args.conv_op in ['mbconv', 'dconv'] else 1,
            stride=block_args.stride,
            bias=False,
        )
        
        out_type = modules['conv1'].out_type  # Update out_type after Conv1
        image_size = calculate_output_image_size(image_size, block_args.stride)

        # Squeeze and Excitation setup
        if self.has_se:
            modules['squeeze'] = EquivariantSqueezeExcitation(
                in_type=out_type,
                sequeeze_ratio=block_args.se_ratio
            )
            out_type = modules['squeeze'].out_type
        
        modules['bn2'] = BatchNorm(in_type=out_type, affine=False)
        modules['swish2'] = Swish(in_type=modules['bn2'].out_type)
        modules['conv2'] = Eq_Conv2dSamePadding(
            in_type=modules['swish2'].out_type,
            out_channels=end_channel_size,
            kernel_size=1 if block_args.conv_op in ['mbconv', 'dconv'] else block_args.kernel_size,
            bias=False,
        )
        
        self.features = nn.Sequential(modules) # use nn.Sequential as we 
        self.dropout = PointwiseDropout(modules['conv2'].out_type, p=dropout_rate)
        self.out_type = self.dropout.out_type

        # Skip connection setup
        if block_args.skip == "conv" or block_args.stride > 1 or self.original_in_type != self.dropout.out_type:
            self.shortcut = SequentialModule(
                BatchNorm(in_type=self.original_in_type, affine=False),
                EquivariantConv(
                    in_type=self.original_in_type,
                    out_channels=len(self.out_type),
                    kernel_size=1,
                    padding=0,
                    stride=block_args.stride,
                    bias=False,
                )
            )
        elif block_args.skip == "identity":
            self.shortcut = nn.Identity()
        elif block_args.skip == "no":
            self.shortcut = None  # No operation
        elif block_args.skip == "pool":
            self.shortcut = EquivariantPool(in_type=self.original_in_type)
        else:
            raise ValueError(f"Unsupported skip connection type: {block_args.skip}")

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.dropout(x)
        
        if self.shortcut is not None:
            x += self.shortcut(inputs)  # Apply skip connection
        return x
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.original_in_type.size
        return input_shape


class NAS_layer(nn.Module):
    def __init__(
            self, 
            in_channel_size: int, 
            block_args: BlockArgs, 
            image_size: int, 
            dropout_rate: float = 0.0,
            expand_ratio: int = 6
        ):
        """
        Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
        """
        super().__init__()
        self.block_args = block_args
        self.original_in_channel_size = in_channel_size
        self.has_se = 0 < block_args.se_ratio <= 1
        layers = OrderedDict()

        self.expand_ratio = expand_ratio if block_args.conv_op == 'mbconv' else 1

        # Expansion phase
        print("in_channel_size", in_channel_size)
        intermedite_channel_size = int(round(in_channel_size * self.expand_ratio))
        if self.expand_ratio != 1:

            layers["bn0"] = nn.BatchNorm2d(num_features=in_channel_size)
            layers["swish0"] = nn.SiLU()
            layers["expand_conv"] = nn.Conv2d(
                in_channels=in_channel_size,
                out_channels=intermedite_channel_size,
                kernel_size=1,
                bias=False,
                padding=get_same_padding(1),
            )

        layers["bn1"] = nn.BatchNorm2d(num_features=intermedite_channel_size)
        layers["swish1"] = nn.SiLU()
        # Conv1
        # potentially depthwise convolution
        groups = intermedite_channel_size if block_args.conv_op in ['mbconv', 'dconv'] else 1
        layers["conv1"] = nn.Conv2d(
            in_channels=intermedite_channel_size,
            out_channels=intermedite_channel_size,
            kernel_size=block_args.kernel_size,
            groups=groups,
            stride=block_args.stride,
            padding=get_same_padding(block_args.kernel_size),
            bias=False,
        )
        image_size = calculate_output_image_size(image_size, block_args.stride)

        # Squeeze and Excitation layer
        if self.has_se:
            num_squeezed_channels = max(1, int(intermedite_channel_size * self.block_args.se_ratio))
            layers["bn_se"] = nn.BatchNorm2d(num_features=intermedite_channel_size)
            layers["se_reduce"] = nn.Conv2d(
                in_channels=intermedite_channel_size,
                out_channels=num_squeezed_channels,
                kernel_size=1,
                padding=get_same_padding(1),
            )
            layers["swish_se"] = nn.SiLU()
            layers["se_expand"] = nn.Conv2d(
                in_channels=num_squeezed_channels,
                out_channels=intermedite_channel_size,
                kernel_size=1,
                padding=get_same_padding(1),
            )

        layers["bn2"] = nn.BatchNorm2d(num_features=intermedite_channel_size)
        layers["swish2"] = nn.SiLU()
        # Conv2
        # potentially pointwise convolution
        kernel_size = 1 if block_args.conv_op in ['mbconv', 'dconv'] else block_args.kernel_size
        layers["conv2"] = nn.Conv2d(
            in_channels=intermedite_channel_size,
            out_channels=block_args.out_channel,
            kernel_size=kernel_size,
            padding=get_same_padding(kernel_size),
            bias=False,
        )
        # transform to nn.Sequential
        self.layers = nn.Sequential(layers)
        self.out_type = block_args.out_channel

        # Skip connection
        if block_args.skip == "conv" \
            or block_args.stride > 1 \
            or self.original_in_channel_size != block_args.out_channel:
            # for larger strides or group changes we have to use a conv layer
            batch_norm = nn.BatchNorm2d(num_features=self.original_in_channel_size)
            shortcut = nn.Conv2d(
                in_channels=self.original_in_channel_size,
                out_channels=block_args.out_channel,
                kernel_size=1,
                padding=0,
                stride=block_args.stride,
                bias=False,
            )
            self.shortcut = nn.Sequential(*[batch_norm, shortcut])
        elif block_args.skip == "identity":
            self.shortcut = nn.Identity()
        elif block_args.skip == "no":
            self.shortcut = None
        elif block_args.skip == "pool":
            raise NotImplementedError("pooling skip connections not implemented yet")
        else:
            raise ValueError(f"Unsupported skip connection type. \
                             Got: {block_args.skip}")


    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        x = self.layers(x)
        
        # Skip connection
        if self.shortcut is not None:
            x += self.shortcut(inputs)
        return x
    
def get_same_padding(kernel_size, stride=1, dilation=1):
    padding = ((kernel_size - 1) * dilation + stride - 1) // 2
    return padding



