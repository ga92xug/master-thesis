import copy
from typing import List, Tuple
from numpy import block
import torch
from torch import batch_norm, nn
from torch.nn import functional as F
import sys

sys.path.append('../networks') # add parent directory

from .util import (
    BlockArgs,
    BlockDecoder,
    get_increase_factor,
    round_repeats,
)
from networks import (
    EquivariantPool, 
    Restriction_from_id,
)
from networks.eq_convs import (
    EquivariantConv,
    EquivariantSqueezeExcitation,
    Eq_Conv2dSamePadding,
    Eq_Conv2dSamePaddingChangeFactor,
)
from networks.gpool_reduction import GroupPoolingReduction

from networks.util import (
    calculate_output_image_size, 
    adjusted_out_channels,
    get_fixed_params,
    get_group_id, 
    get_gspace_from_id, 
    get_param_count,
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

class Eq_NAS_Block(EquivariantModule):
    """
    Block with variable content based on block_args.
    """
    def __init__(
            self, 
            in_type: FieldType, 
            in_channel_size, 
            fixed_params, 
            block_args, 
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
        
        intermedite_channel_size = adjusted_out_channels(
                out_channel=in_channel_size,
                N=in_type.gspace.fibergroup.order(),
                fixed_params=fixed_params,
            )
        end_channel_size = adjusted_out_channels(
                out_channel=block_args.out_channel,
                N=in_type.gspace.fibergroup.order(),
                fixed_params=fixed_params,
            )

        # Expansion phase
        self.expand_ratio = expand_ratio if block_args.conv_op == 'mbconv' else 1
        intermedite_channel_size = int(round(intermedite_channel_size * self.expand_ratio))
        if self.expand_ratio != 1:
            self._bn0 = BatchNorm(in_type=in_type, affine=False)
            self._swish0 = Swish(in_type=self._bn0.out_type)
            self._expand_conv = Eq_Conv2dSamePadding(
                in_type=self._swish0.out_type,
                out_channels=intermedite_channel_size,
                kernel_size=1,
                bias=False,
            )
            in_type = self._expand_conv.out_type

        self._bn1 = BatchNorm(in_type=in_type, affine=False)
        self._swish1 = Swish(in_type=self._bn1.out_type)
        # Conv1
        # potentially depthwise convolution
        if block_args.conv_op == 'dconv' and intermedite_channel_size % len(in_type) != 0:
            # depthwise convolution only if no restriction
            # otherwise the input size is not divisible by the number of groups
            block_args = block_args._replace(conv_op='conv')

        groups = len(self._swish1.out_type) if block_args.conv_op in ['mbconv', 'dconv'] else 1
        self._conv1 = Eq_Conv2dSamePadding(
            in_type=self._swish1.out_type,
            out_channels=intermedite_channel_size,
            kernel_size=block_args.kernel_size,
            groups=groups,
            stride=block_args.stride,
            bias=False,
        )
        out_type = self._conv1.out_type
        image_size = calculate_output_image_size(image_size, block_args.stride)

        # Squeeze and Excitation layer
        if self.has_se:
            self._squeeze = EquivariantSqueezeExcitation(
                in_type=out_type,
                sequeeze_ratio=block_args.se_ratio
            )
            out_type = self._squeeze.out_type

        self._bn2 = BatchNorm(in_type=out_type, affine=False)
        self._swish2 = Swish(in_type=self._bn2.out_type)
        # Conv2
        # potentially pointwise convolution
        kernel_size = 1 if block_args.conv_op in ['mbconv', 'dconv'] else block_args.kernel_size
        self._conv2 = Eq_Conv2dSamePadding(
            in_type=self._swish2.out_type,
            out_channels=end_channel_size,
            kernel_size=kernel_size,
            bias=False,
        )
        self.dropout = PointwiseDropout(self._conv2.out_type, p=dropout_rate)
        self.out_type = self.dropout.out_type

        #print("skip connection ", self.original_in_type.size, self._conv2.out_type.size)
        #print("type ", self.original_in_type, self._conv2.out_type)
        # Skip connection
        # if block_args.skip == "identity" and block_args.stride == 1 and \
        #     in_channel_size == block_args.out_channel:
        #     #print("skip identity")
        #     self.shortcut = GroupPoolingReduction(self.original_in_type)

        if block_args.skip == "conv" \
            or block_args.stride > 1 \
            or self.original_in_type != self.dropout.out_type:
            #print("skip conv")
            # for larger strides or group changes we have to use a conv layer
            batch_norm = BatchNorm(in_type=self.original_in_type, affine=False)
            shortcut = EquivariantConv(
                in_type=batch_norm.out_type,
                out_channels=len(self.out_type),
                kernel_size=1,
                padding=0,
                stride=block_args.stride,
                bias=False,
            )
            self.shortcut = SequentialModule(*[batch_norm, shortcut])
        elif block_args.skip == "identity":
            self.shortcut = nn.Identity()
        elif block_args.skip == "no":
            self.shortcut = None
        elif block_args.skip == "pool":
            self.shortcut = EquivariantPool(
                in_type=self.original_in_type,
            )
        else:
            raise ValueError(f"Unsupported skip connection type. \
                             Got: {block_args.skip}")

        #print()
        
    def forward(self, inputs):
        x = inputs
        # Expansion
        if self.expand_ratio != 1:
            x = self._bn0(x)
            x = self._swish0(x)
            x = self._expand_conv(x)
            
        x = self._bn1(x)
        x = self._swish1(x)
        x = self._conv1(x)

        # Squeeze and Excitation
        if self.has_se:
            x = self._squeeze(x)

        # Pointwise Convolution
        x = self._bn2(x)
        x = self._swish2(x)
        x = self._conv2(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Skip connection
        if self.shortcut is not None:
            # print("x", x.type)
            # print("shortcut", self.shortcut(inputs).type)
            # print("inputs", inputs.type)
            x = x + self.shortcut(inputs) # skip connection
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.original_in_type.size
        return input_shape


class NAS_Block(nn.Module):
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

        self.expand_ratio = expand_ratio if block_args.conv_op == 'mbconv' else 1

        # Expansion phase
        print("in_channel_size", in_channel_size)
        intermedite_channel_size = int(round(in_channel_size * self.expand_ratio))
        if self.expand_ratio != 1:
            self._bn0 = nn.BatchNorm2d(num_features=in_channel_size)
            self._swish0 = nn.SiLU()
            self._expand_conv = Conv2dSamePadding(
                in_channels=in_channel_size,
                out_channels=intermedite_channel_size,
                kernel_size=1,
                bias=False,
            )

        self._bn1 = nn.BatchNorm2d(num_features=intermedite_channel_size)
        self._swish1 = nn.SiLU()
        # Conv1
        # potentially depthwise convolution
        groups = intermedite_channel_size if block_args.conv_op in ['mbconv', 'dconv'] else 1
        self._conv1 = Conv2dSamePadding(
            in_channels=intermedite_channel_size,
            out_channels=intermedite_channel_size,
            kernel_size=block_args.kernel_size,
            groups=groups,
            stride=block_args.stride,
            bias=False,
        )
        image_size = calculate_output_image_size(image_size, block_args.stride)

        # Squeeze and Excitation layer
        if self.has_se:
            num_squeezed_channels = max(1, int(intermedite_channel_size * self.block_args.se_ratio))
            self._bn_se = nn.BatchNorm2d(num_features=intermedite_channel_size)
            self._se_reduce = Conv2dSamePadding(in_channels=intermedite_channel_size, out_channels=num_squeezed_channels, kernel_size=1)
            self._swish_se = nn.SiLU()
            self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels, out_channels=intermedite_channel_size, kernel_size=1)


        self._bn2 = nn.BatchNorm2d(num_features=intermedite_channel_size)
        self._swish2 = nn.SiLU()
        # Conv2
        # potentially pointwise convolution
        kernel_size = 1 if block_args.conv_op in ['mbconv', 'dconv'] else block_args.kernel_size
        self._conv2 = Conv2dSamePadding(
            in_channels=intermedite_channel_size,
            out_channels=block_args.out_channel,
            kernel_size=kernel_size,
            bias=False,
        )
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
        if self.expand_ratio != 1:
            x = self._bn0(inputs)
            x = self._swish0(x)
            x = self._expand_conv(x)

        x = self._bn1(x)
        x = self._swish1(x)
        x = self._conv1(x)

        # Squeeze and Excitation
        if self.has_se:
            x = self._bn_se(x)
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish_se(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._bn2(x)
        x = self._swish2(x)
        x = self._conv2(x)
        

        # Skip connection
        if self.shortcut is not None:
            x = self.shortcut(inputs) + x
        return x
    

PADDINGS = {
    1: 0,
    3: 1,
    5: 2,
    7: 3,
}

class Conv2dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, groups=1) -> None:
        super().__init__()
        padding = PADDINGS[kernel_size]
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)



