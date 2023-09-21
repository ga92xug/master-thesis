import math
import re
from numpy import block
from requests import get
import wandb
from typing import List, Tuple
from matplotlib.pyplot import stem
from torch import nn
from omegaconf import DictConfig, OmegaConf
import sys
import os
sys.path.append(f"{os.getcwd()}")

from .util import (
    BlockArgs,
    BlockDecoder,
    get_increase_factor,
    round_repeats,
    get_channel_sizes,
)
from networks.eq_nasnet.util import encode_parameters
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
    adjusted_out_channels,
    compare_dicts,
    flatten_dict,
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


def get_blocks_args(blocks_args_new:dict):
    [BlockArgs(**block_args_new) for block_args_new in blocks_args_new.values()]
    blocks_args = []
    for level, block_args_new in blocks_args_new.items():
        block_args = BlockArgs(**block_args_new)
        blocks_args.append(block_args)
    return blocks_args


class EquivariantNASNet(nn.Module):
    def __init__(
            self, 
            blocks_args_dict,
            #blocks_args, 
            image_size,
            width_coefficient=1, 
            depth_coefficient=1,
            dropout_rate=0.2,
            stem_channels=16,
            fixed_params=True,
            eq_expand_ratio=2,
            cnn_expand_ratio=6,
            input_channels=3, 
            num_classes=10, 
            **kwargs,
    ):
        print("Equivariant_NAS_Net")
        super().__init__()        
        #blocks_args = list(blocks_args)
        assert image_size is not None, 'Please provide image size'
        #assert isinstance(blocks_args, list), f'blocks_args should be a list, is a {type(blocks_args)}'
        #assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.dropout_rate = dropout_rate
        self.eq_expand_ratio = eq_expand_ratio
        self.cnn_expand_ratio = cnn_expand_ratio
        self.fixed_params = fixed_params

        # BlockArgs
        print("block_args_dict", blocks_args_dict)
        blocks_args = [BlockArgs(**block_args_dict) for block_args_dict in blocks_args_dict.values()]
        BlockDecoder()._check_valid_blocks_args(blocks_args)
        
        # The channel sizes is first an increase factor. After that it is the number of channels
        self.blocks_args = get_channel_sizes(stem_channels, blocks_args, width_coefficient)
        stem_args = blocks_args[0]

        # Get group spaces for specified rotations and flips
        group_id = get_group_id(stem_args.reflection, stem_args.group)
        gspace = get_gspace_from_id(group_id)
        self.gspace = gspace

        self.set_name()

        image_size = [image_size]*2 if isinstance(image_size, int) else image_size
        
        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * input_channels
        )

        # Stem
        print("Building stem")
        channel_size = adjusted_out_channels(
                out_channel = stem_args.out_channel,
                N=self.gspace.fibergroup.order(),
                fixed_params=fixed_params,
            )
        self._conv_stem = Eq_Conv2dSamePadding(
            in_type=self.input_field_type,
            out_channels=channel_size,
            kernel_size=stem_args.kernel_size,
            stride=stem_args.stride,
            bias=False,
        )
        self.field_type = self._conv_stem.out_type
        self.prev_channel_size = stem_args.out_channel
        image_size = calculate_output_image_size(image_size, stem_args.stride)

        # Build blocks
        self._blocks = nn.ModuleList([])
        # block 0 is the stem, block -1 is the head
        for i, block_args in enumerate(self.blocks_args[1:-1]):
            print(f"Building block: {i+1}")
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                num_layers=round_repeats(
                    block_args.num_layers, 
                    depth_coefficient
                ),
            )
            group_id = get_group_id(block_args.reflection, block_args.group)
            restrict = Restriction_Group_or_CNN(self.field_type,group_id)
            self._blocks.append(restrict)
            self.field_type = restrict.out_type
            # The first block needs to take care of stride and filter size increase.
            block = self.create_block(restrict, block_args, image_size)
            self._blocks.append(block)
            image_size = calculate_output_image_size(image_size, 
                                                     block_args.stride)
            if block_args.num_layers > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(stride=1)
            
            for _ in range(block_args.num_layers - 1):
                block = self.create_block(restrict, block_args, image_size)
                self._blocks.append(block)


        last_block_args = self.blocks_args[-1]
        group_id = get_group_id(last_block_args.reflection, last_block_args.group)
        self.restrict_last = Restriction_Group_or_CNN(self.field_type, group_id)
        if last_block_args.kernel_size != 0:
            # Build head
            print("Building head")
            # Restrict
            self.field_type = self.restrict_last.out_type
        
        # Head
        if self.restrict_last.setting in ["cnn", "switch"]:
            N = 0
        else:
            N = self.gspace.fibergroup.order()
        out_channels = adjusted_out_channels(
                out_channel = last_block_args.out_channel,
                N=N,
            )
        
        if self.restrict_last.setting in ["cnn", "switch"]:
            self._bn1 = nn.BatchNorm2d(self.field_type)
            self._swish1 = nn.SiLU()
            self._conv_head = Conv2dSamePadding(
                in_channels=self.field_type,
                out_channels=out_channels,
                kernel_size=last_block_args.kernel_size,
                bias=False,
            )
            self._bn2 = nn.BatchNorm2d(out_channels)
            self._swish2 = nn.SiLU()
            
        elif self.restrict_last.setting == "group": 
            if last_block_args.kernel_size != 0:
                self._bn1 = BatchNorm(in_type=self.field_type, affine=False)
                self._swish1 = Swish(in_type=self._bn1.out_type)
                self._conv_head = Eq_Conv2dSamePadding(
                    in_type=self._swish1.out_type, 
                    out_channels=out_channels,
                    bias=False
                )
                self.field_type = self._conv_head.out_type
            
            self._bn2 = BatchNorm(in_type=self.field_type, affine=False)
            self._swish2 = Swish(in_type=self._bn2.out_type)
            # Final linear layer
            self.invariant_map = EquivariantPool(
                self._swish2.out_type, 
                invariant_map=True
            )
        else:
            raise NotImplementedError(f"This setting: {self.restrict_last.setting} is not implemented")
        # pooling
        print("pooling image size: ", image_size)
        #assert image_size[0] <= 8, "We don't want to pool too much, check num_blocks"
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(self.dropout_rate)
        if self.restrict_last.setting in ["cnn", "switch"]:
            self.fc = nn.Linear(out_channels, num_classes)
        else:
            self.fc = nn.Linear(len(self.invariant_map.out_type), num_classes)


    def forward(self, inputs):
        x = GroupTensor(inputs, self.input_field_type)
        # Stem
        x = self._conv_stem(x)
        # Blocks
        for idx, restrict_or_MBBlock in enumerate(self._blocks):
            # if isinstance(restrict_or_MBBlock, Eq_NAS_Block):
            #     print(f"Running block: {idx}")
            x = restrict_or_MBBlock(x)

        # Head
        if self.blocks_args[-1].kernel_size != 0:
            x = self.restrict_last(x)
            x = self._conv_head(self._swish1(self._bn1(x)))

        # final batch norm and swish
        x = self._swish2(self._bn2(x))

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
        if setting in ["cnn", "switch"]:
            
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
                    in_channel_size=self.prev_channel_size,
                    fixed_params=self.fixed_params, 
                    block_args=block_args, 
                    image_size=image_size,
                    dropout_rate=self.dropout_rate,
                    expand_ratio=self.eq_expand_ratio,
                )
        self.prev_channel_size = block_args.out_channel
        self.field_type = block.out_type
            
        return block

    
    def set_name(self):
        self.name = f"eq_nasnet" 

        self.name += f"{self.gspace.fibergroup}"
        print(self.name)

