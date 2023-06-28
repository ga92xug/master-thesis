import math
import re
from typing import List, Tuple
from matplotlib.pyplot import stem
from torch import nn
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('../networks') # add parent directory

from .util import (
    BlockArgs,
    BlockDecoder,
    get_increase_factor,
    get_out_channels,
    round_repeats,
)
from networks.eq_restriction import Restriction_Group_or_CNN
from networks.eq_nasnet.nas_block import Conv2dSamePadding, Eq_NAS_Block, NAS_Block

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
            eq_expand_ratio=2,
            cnn_expand_ratio=6,
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
        self.eq_expand_ratio = eq_expand_ratio
        self.cnn_expand_ratio = cnn_expand_ratio
        # BlockArgs
        blocks_args = BlockDecoder.decode(blocks_args)
        self.channel_sizes = self.get_channel_sizes(stem_channels, blocks_args)
        print(f"channel_sizes: {self.channel_sizes}")
        self.blocks_args = blocks_args
        stem_args = blocks_args[0]

        # Get group spaces for specified rotations and flips
        self.reflection = stem_args.reflection
        self.group = stem_args.group
        group_id = get_group_id(self.reflection, self.group)
        gspace = get_gspace_from_id(group_id)
        self.gspace = gspace

        image_size = [image_size]*2 if isinstance(image_size, int) else image_size
        

        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * input_channels
        )

        # Stem
        print("Building stem")
        out_channels = get_out_channels(
            in_type=stem_channels,
            increase_factor=get_increase_factor(
                stem_args.channel_increase_factor, 
                self.width_coefficient,
                self.depth_divisor, 
                self.min_depth
            ), 
            new_rotation=stem_args.group,
            is_first=True,
        )
        self._conv_stem = Eq_Conv2dSamePadding(
            in_type=self.input_field_type,
            out_channels=out_channels,
            kernel_size=stem_args.kernel_size,
            stride=stem_args.stride,
            bias=False,
        )
        self._bn0 = BatchNorm(in_type=self._conv_stem.out_type)
        self._swish0 = Swish(in_type=self._bn0.out_type)
        self.field_type = self._swish0.out_type
        image_size = calculate_output_image_size(image_size, stem_args.stride)

        # Build blocks
        self._blocks = nn.ModuleList([])
        # we start with the first block 
        # block 0 is the stem
        for i, block_args in enumerate(self.blocks_args[1:-1]):
            print(f"Building block: {i+1}")
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                channel_increase_factor=get_increase_factor(
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
            restrict = Restriction_Group_or_CNN(self.field_type,group_id)
            self._blocks.append(restrict)
            self.field_type = restrict.out_type
            # The first block needs to take care of stride and filter size increase.
            block = self.create_block(restrict, block_args, image_size)
            self._blocks.append(block)
            self.field_type = self._blocks[-1].out_type
            image_size = calculate_output_image_size(image_size, 
                                                     block_args.stride)
            if block_args.num_layers > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(stride=1, channel_increase_factor=1)
            
            for _ in range(block_args.num_layers - 1):
                block = self.create_block(restrict, block_args, image_size)
                self._blocks.append(block)
                self.field_type = self._blocks[-1].out_type


        # Build head
        print("Building head")
        last_block_args = self.blocks_args[-1]
        # Restrict
        group_id = get_group_id(last_block_args.reflection, last_block_args.group)
        self.restrict_last = Restriction_Group_or_CNN(self.field_type, group_id)
        #restriction_correction_factor = self.restrict_last.get_correction_factor()
        self.field_type = self.restrict_last.out_type

        # Head
        channel_increase_factor = get_increase_factor(
            last_block_args.channel_increase_factor, 
            self.width_coefficient,
            self.depth_divisor, 
            self.min_depth
        )
        out_channels = get_out_channels(
                in_type=self.field_type, 
                increase_factor=channel_increase_factor,
                new_rotation=block_args.group,
            )
        
        if self.restrict_last.setting in ["cnn", "switch"]:
            self._conv_head = Conv2dSamePadding(
                in_channels=self.field_type,
                out_channels=out_channels,
                kernel_size=last_block_args.kernel_size,
                bias=False,
            )
            self._bn1 = nn.BatchNorm2d(out_channels)
            self._swish1 = nn.SiLU()

        elif self.restrict_last.setting == "group": 
            self._conv_head = Eq_Conv2dSamePadding(
                in_type=self.field_type, 
                out_channels=out_channels,
                bias=False
            )
            self._bn1 = BatchNorm(in_type=self._conv_head.out_type)
            self._swish1 = Swish(in_type=self._bn1.out_type)

            # Final linear layer
            self.invariant_map = EquivariantPool(
                self._swish1.out_type, 
                invariant_map=True
            )
        else:
            raise NotImplementedError(f"This setting: {self.restrict_last.setting} is not implemented")
        # pooling
        print("pooling image size: ", image_size)
        #assert image_size[0] <= 8, "We don't want to pool too much, check num_blocks"
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(len(self._swish1.out_type), num_classes)


    def forward(self, inputs):
        x = GroupTensor(inputs, self.input_field_type)

        # Stem
        x = self._swish0(self._bn0(self._conv_stem(x)))
        # Blocks
        for idx, restrict_or_MBBlock in enumerate(self._blocks):
            
            # if isinstance(restrict_or_MBBlock, Eq_NAS_Block):
            #     print(f"Running block: {idx}")
            x = restrict_or_MBBlock(x)

        # Head
        x = self.restrict_last(x)
        x = self._swish1(self._bn1(self._conv_head(x)))
        # transfer to invariant if not already
        if hasattr(self, "invariant_map"):
            x = self.invariant_map(x)
            x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


    def create_block(
        self,
        restrict: nn.Module,
        block_args: BlockArgs,
        image_size: Tuple[int, int],
    ):  
        setting = restrict.setting
        if setting in ["CNN", "switch"]:
            
            block = NAS_Block(
                    in_channel_size=self.field_type,
                    block_args=block_args,
                    image_size=image_size,
                    dropout_rate=self.dropout_rate,
                    expand_ratio=self.cnn_expand_ratio,
                )
        
        else:
            block = Eq_NAS_Block(
                    in_type=self.field_type, 
                    block_args=block_args, 
                    image_size=image_size,
                    dropout_rate=self.dropout_rate,
                    expand_ratio=self.eq_expand_ratio,
                )
            
        return block

    def get_channel_sizes(self, initial_channel_size, blocks_args):
        channel_sizes = [initial_channel_size]
        for block_args in blocks_args:
            increase_factor = get_increase_factor(
                block_args.channel_increase_factor, 
                self.width_coefficient,
                self.depth_divisor, 
                self.min_depth
            )
            initial_channel_size.append(initial_channel_size[-1] * increase_factor)
        return channel_sizes