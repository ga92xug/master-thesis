import math
from typing import List, Tuple
from torch import nn
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('../networks') # add parent directory

from .util import (
    BlockDecoder,
    eq_round_filters,
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

from networks.util import (
    calculate_output_image_size, 
    get_fixed_params,
    get_group_id, 
    get_gspace_from_id, 
    get_param_count,
)

from nn import (
    GroupTensor,
    FieldType,
    EquivariantModule,
    BatchNorm,
    Mish,
    ReLU,
    Swish,
)

import os
os.environ['HYDRA_FULL_ERROR'] = '1'


class Eq_NAS_Block(EquivariantModule):
    """
    Block with variable content based on block_args.
    """
    def __init__(self, in_type, block_args, image_size, 
                 restriction_correction_factor=1, dropout_rate=0.0,
                 expand_ratio=2):
        """
        Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
        """
        super().__init__()
        self.block_args = block_args
        self.original_in_type = in_type
        self.has_se = 0 < block_args.se_ratio <= 1

        self.expand_ratio = expand_ratio if block_args.conv_op == 'mbconv' else 1

        # Expansion phase
        if self.expand_ratio != 1:
            self._expand_conv = Eq_Conv2dSamePaddingChangeFactor(
                in_type=in_type,
                change_factor=self.expand_ratio,
                kernel_size=1,
                bias=False,
            )
            self._bn0 = BatchNorm(in_type=self._expand_conv.out_type)
            self._swish0 = Swish(in_type=self._bn0.out_type)
            in_type = self._swish0.out_type

        # Conv1
        # potentially depthwise convolution
        groups = len(in_type) if block_args.conv_op in ['mbconv', 'dconv'] else 1
        self.conv1 = Eq_Conv2dSamePaddingChangeFactor(
            in_type=in_type,
            change_factor=restriction_correction_factor,
            kernel_size=block_args.kernel_size,
            groups=groups,
            stride=block_args.stride,
            bias=False,
        )
        self.bn1 = BatchNorm(in_type=self.conv1.out_type)
        self.swish1 = Swish(in_type=self.bn1.out_type)
        out_type = self.swish1.out_type
        image_size = calculate_output_image_size(image_size, block_args.stride)

        # Squeeze and Excitation layer
        if self.has_se:
            self.squeeze = EquivariantSqueezeExcitation(
                in_type=out_type,
                sequeeze_ratio=block_args.se_ratio
            )
            out_type = self.squeeze.out_type

        # Conv2
        # potentially pointwise convolution
        kernel_size = 1 if block_args.conv_op in ['mbconv', 'dconv'] else block_args.kernel_size
        self.conv2 = Eq_Conv2dSamePaddingChangeFactor(
            in_type=out_type,
            change_factor=block_args.channel_increase_factor,
            kernel_size=kernel_size,
            bias=False,
        )
        self.bn2 = BatchNorm(in_type=self.conv2.out_type)
        self.out_type = self.bn2.out_type

        # Skip connection
        if block_args.skip == "conv" \
            or block_args.stride > 1 \
            or self.in_type != self.out_type:
            # for larger strides or group changes we have to use a conv layer
            self.shortcut = EquivariantConv(
                in_type=self.original_in_type,
                out_channels=len(self.bn2.out_type),
                kernel_size=1,
                padding=0,
                stride=block_args.stride,
                bias=False,
            )
        elif block_args.skip == "identity":
            self.shortcut = nn.Identity()
        elif block_args.skip == "no":
            self.shortcut = None
        elif block_args.skip == "pool":
            self.shortcut = EquivariantPool(
                in_type=self.original_in_type,
                stride=block_args.stride,
            )
        else:
            raise ValueError(f"Unsupported skip connection type. \
                             Got: {block_args.skip}")

        
    def forward(self, inputs):
        x = inputs
        # Expansion
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._swish0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish1(x)

        # Squeeze and Excitation
        if self.has_se:
            x = self.squeeze(x)

        # Pointwise Convolution
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Skip connection
        if self.shortcut is not None:
            #print("x", x.tensor.shape, "shortcut", shortcut_result.tensor.shape)
            x = x + self.shortcut(inputs) # skip connection
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.original_in_type.size
        return input_shape


class EquivariantNASNet(nn.Module):
    def __init__(
            self, 
            blocks_args, 
            image_size,
            width_coefficient=1, 
            depth_coefficient=1,
            dropout_rate=0.2,
            depth_divisor=8,
            min_depth=1,
            stem_channels=16,
            expand_ratio=2,
            input_channels=3, 
            num_classes=10, 
    ):
        print("Equivariant_NAS_Net")
        super().__init__()        
        blocks_args = list(blocks_args)
        assert image_size is not None, 'Please provide image size'
        assert isinstance(blocks_args, list), f'blocks_args should be a list, is a {type(blocks_args)}'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.expand_ratio = expand_ratio
        # BlockArgs
        blocks_args = BlockDecoder.decode(blocks_args)
        self.blocks_args = blocks_args
        stem_args = blocks_args[0]

        # Get group spaces for specified rotations and flips
        self.reflection = stem_args.reflection
        self.group = stem_args.group
        group_id = get_group_id(self.reflection, self.group)
        gspace = get_gspace_from_id(group_id)
        self.gspace = gspace

        self.input_channels = input_channels
        image_size = [image_size]*2 if isinstance(image_size, int) else image_size
        self.image_size = image_size
        self.num_classes = num_classes
        

        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.input_channels
        )

        # Stem
        print("Building stem")
        out_channels = (stem_channels / stem_args.group) \
            * math.sqrt(stem_args.group)
        channel_increase_factor = eq_round_filters(
            stem_args.channel_increase_factor, 
            self.width_coefficient, 
            self.depth_divisor, 
            self.min_depth
        )
        self._conv_stem = Eq_Conv2dSamePadding(
            in_type=self.input_field_type,
            out_channels=int(out_channels * channel_increase_factor),
            kernel_size=stem_args.kernel_size,
            stride=2,
            bias=False,
        )
        self._bn0 = BatchNorm(in_type=self._conv_stem.out_type)
        self._swish0 = Swish(in_type=self._bn0.out_type)

        self.field_type = self._swish0.out_type
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        # we start with the first block 
        # block 0 is the stem
        for i, block_args in enumerate(self.blocks_args[1:-1]):
            print(f"Building block: {i+1}")
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                channel_increase_factor=eq_round_filters(
                    block_args.channel_increase_factor, 
                    self.width_coefficient, 
                    self.depth_divisor, 
                    self.min_depth
                ),
                num_layers=round_repeats(
                    block_args.num_layers, 
                    self.depth_coefficient
                ),
            )
            group_id = get_group_id(block_args.reflection, block_args.group)
            restrict = Restriction_from_id(self.field_type,group_id)
            self._blocks.append(restrict)
            self.field_type = restrict.out_type
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                Eq_NAS_Block(
                        self.field_type, 
                        block_args, 
                        image_size=image_size,
                        restriction_correction_factor=restrict.get_correction_factor(),
                        dropout_rate=self.dropout_rate,
                        expand_ratio=self.expand_ratio,
                    )
            )
            self.field_type = self._blocks[-1].out_type
            image_size = calculate_output_image_size(image_size, 
                                                     block_args.stride)
            if block_args.num_layers > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(stride=1, channel_increase_factor=1)
            for _ in range(block_args.num_layers - 1):
                self._blocks.append(
                    Eq_NAS_Block(
                        in_type=self.field_type, 
                        block_args=block_args, 
                        image_size=image_size, 
                        dropout_rate=self.dropout_rate,
                        expand_ratio=self.expand_ratio
                    )
                )
                self.field_type = self._blocks[-1].out_type


        # Build head
        print("Building head")
        last_block_args = self.blocks_args[-1]
        # Restrict
        group_id = get_group_id(last_block_args.reflection, last_block_args.group)
        self.restrict_last = Restriction_from_id(self.field_type, group_id)
        restriction_correction_factor = self.restrict_last.get_correction_factor()
        self.field_type = self.restrict_last.out_type

        # Head
        channel_increase_factor = eq_round_filters(
            last_block_args.channel_increase_factor, 
            self.width_coefficient,
            self.depth_divisor, 
            self.min_depth
        )
        self._conv_head = Eq_Conv2dSamePaddingChangeFactor(
            in_type=self.field_type, 
            change_factor=channel_increase_factor * restriction_correction_factor, 
            bias=False
        )
        self._bn1 = BatchNorm(in_type=self._conv_head.out_type)
        self._swish1 = Swish(in_type=self._bn1.out_type)

        # Final linear layer
        self.invariant_map = EquivariantPool(
            self._swish1.out_type, 
            invariant_map=True
        )

        # pooling
        assert image_size[0] <= 8, "We don't want to pool too much, check num_blocks"
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        print("image size: ", image_size)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(len(self._swish1.out_type), self.num_classes)


    def forward(self, inputs):
        x = GroupTensor(inputs, self.input_field_type)

        # Stem
        x = self._swish0(self._bn0(self._conv_stem(x)))
        # Blocks
        for idx, restrict_or_MBBlock in enumerate(self._blocks):
            x = restrict_or_MBBlock(x)

        # Head
        x = self.restrict_last(x)
        x = self._swish1(self._bn1(self._conv_head(x)))
        # Pooling and final linear layer
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
