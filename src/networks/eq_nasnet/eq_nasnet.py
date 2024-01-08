from typing import Tuple, Callable, Iterable, List, Dict, Any

from torch import nn
from omegaconf import DictConfig, OmegaConf
import sys
import os
from src.networks.eq_nasnet.block_args import BlockArgsList

from src.networks.eq_nasnet.naming_eq_nasnet import get_scaling_name
sys.path.append(f"{os.getcwd()}")

from networks.eq_restriction import Restriction_Group_or_CNN
from networks.eq_nasnet.nas_block import Conv2dSamePadding, Eq_NAS_layer, NAS_layer

from networks import (
    EquivariantPool, 
)
from networks.eq_convs import (
    Eq_Conv2dSamePadding,
)

from networks.util import (
    get_group_id, 
    get_gspace_from_id, 
    adjusted_out_channels,
)

from equivariant.nn import (
    GroupTensor,
    FieldType,
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
        blocks_args_dict: dict,
        image_size: int,
        width_coefficient=1, 
        depth_coefficient=1,
        dropout_rate=0.2,
        stem_channels=16,
        fixed_params=False,
        eq_expand_ratio=2,
        cnn_expand_ratio=6,
        num_channels=3, 
        num_classes=10,
        not_increase_1_layer=True,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__()        
        #blocks_args = list(blocks_args)
        assert isinstance(image_size, int), 'Please provide valid image size'
        self.image_size = image_size
        #assert isinstance(blocks_args, list), f'blocks_args should be a list, is a {type(blocks_args)}'
        #assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.verbose = verbose
        self.dropout_rate = dropout_rate
        self.eq_expand_ratio = eq_expand_ratio
        self.cnn_expand_ratio = cnn_expand_ratio
        self.depth_coefficient = depth_coefficient
        self.width_coefficient = width_coefficient
        self.fixed_params = fixed_params

        # BlockArgs
        self.blocks_args_list = BlockArgsList.from_dict(blocks_args_dict, stem_channels, width_coefficient, depth_coefficient)

        
        stem_args = self.blocks_args_list[0]
        self.set_name()

        # Get group spaces for specified rotations and flips
        group_id = get_group_id(stem_args.reflection, stem_args.group)
        gspace = get_gspace_from_id(group_id)
        self.gspace = gspace

        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * num_channels
        )

        # Stem
        if verbose > 3:
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
        image_size = int(math.ceil(image_size / stem_args.stride))

        # Build blocks
        self._blocks = nn.ModuleDict({})
        
        # block 0 is the stem, block -1 is the head
        for i, block_args in enumerate(self.blocks_args_list[1:-1]):
            if self.verbose > 3:
                print(f"Building block: {i+1}")
            self._blocks[f"block{i+1}"] = self.create_block(block_args)

        last_block_args = self.blocks_args_list[-1]
        group_id = get_group_id(last_block_args.reflection, last_block_args.group)
        self.restrict_last = Restriction_Group_or_CNN(self.field_type, group_id)
        if last_block_args.kernel_size != 0:
            # Build head
            if verbose > 3:
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
        if verbose > 3:
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
        for _, block in self._blocks:
            for i, restrict_or_MBBlock in enumerate(block):
            # if isinstance(restrict_or_MBBlock, Eq_NAS_Block):
            #     print(f"Running block: {i}")
                x = restrict_or_MBBlock(x)

        # Head
        if self.blocks_args_list[-1].kernel_size != 0:
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
    
    def create_block(self, block_args) -> nn.Module:
        layers = nn.ModuleList([])

        # restriction
        group_id = get_group_id(block_args.reflection, self.block_args.group)
        restrict = Restriction_Group_or_CNN(self.field_type,group_id)
        layers.append(restrict)
        self.field_type = restrict.out_type

        # The first layer needs to take care of stride and filter size increase.
        layers.append(self.create_conv_layer(restrict, block_args, image_size))
        image_size = int(math.ceil(image_size / self.block_args.stride))
        
        # Add rest of layers
        block_args = block_args._replace(stride=1)
        for _ in range(block_args.num_layers - 1):
            layers.append(
                self.create_conv_layer(restrict, block_args, image_size)
            )

        return layers


    def create_conv_layer(
        self,
        restrict: nn.Module,
        block_args: BlockArgs,
        image_size: int,
    )-> nn.Module:  
        setting = restrict.setting
        if setting in ["cnn", "switch"]:
            
            layer = NAS_layer(
                    in_channel_size=self.field_type,
                    block_args=block_args,
                    image_size=image_size,
                    dropout_rate=self.dropout_rate,
                    expand_ratio=self.cnn_expand_ratio,
                )
        
        else:
            layer = Eq_NAS_layer(
                    in_type=self.field_type,
                    in_channel_size=self.prev_channel_size,
                    fixed_params=self.fixed_params, 
                    block_args=block_args, 
                    image_size=image_size,
                    dropout_rate=self.dropout_rate,
                    expand_ratio=self.eq_expand_ratio,
                )
        self.prev_channel_size = block_args.out_channel
        self.field_type = layer.out_type
            
        return layer

    
    def set_name(self):
        model_name = f"eq_nasnet_{self.gspace.fibergroup}_b{len(self.blocks_args_list)-2}_\
            d{self.depth_coefficient}_w{self.width_coefficient}_\
            r{self.image_size[0]}_drop{self.dropout_rate}"
        scaling_name = get_scaling_name(self.blocks_args_list)
        self.name = {
            "model_name": model_name,
            "scaling_name": scaling_name,
        }


