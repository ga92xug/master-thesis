from typing import Tuple, Callable, Iterable, List, Dict, Any, Mapping
import math
from torch import nn
import sys
import os
import torch
sys.path.append(f"{os.getcwd()}")
from src.networks.eq_nasnet.utils import pool_like, set_eq_nasnet_name
from src.utils.equivariant_utils import create_filters_network
from src.networks.eq_nasnet.block_args import BlockArgs, BlockArgsList
from src.networks.equivariant_utils.eq_restriction import Restriction_Group_or_CNN
from src.networks.eq_nasnet.nas_block import Conv2dSamePadding, Eq_NAS_layer, NAS_layer
from src.networks import EquivariantPool
from src.networks.equivariant_utils.eq_convs import Eq_Conv2dSamePadding
from src.networks.equivariant_utils.utils import (
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

class EquivariantNASNet(nn.Module):
    def __init__(
        self, 
        blocks_args_dict: Dict[str, Any],
        image_size: int,
        pre_trained: str = None,
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
        assert isinstance(image_size, int), 'Please provide valid image size'
        self.pre_trained = pre_trained
        self.image_size = image_size
        self.verbose = verbose
        self.dropout_rate = dropout_rate
        self.eq_expand_ratio = eq_expand_ratio
        self.cnn_expand_ratio = cnn_expand_ratio
        self.depth_coefficient = depth_coefficient
        self.width_coefficient = width_coefficient
        self.fixed_params = fixed_params
        self.num_channels = num_channels

        # BlockArgs
        self.blocks_args_list = BlockArgsList.from_dict(blocks_args_dict, stem_channels, width_coefficient, depth_coefficient)
        stem_args = self.blocks_args_list[0]
        
        # Get group spaces for specified rotations and flips
        group_id = get_group_id(stem_args.reflection, stem_args.group)
        gspace = get_gspace_from_id(group_id)
        self.gspace = gspace

        self.name = set_eq_nasnet_name(
            pre_trained=pre_trained,
            gspace=gspace,
            blocks_args_list=self.blocks_args_list,
            depth_coefficient=depth_coefficient,
            width_coefficient=width_coefficient,
            image_size=image_size,
            dropout_rate=dropout_rate,
        )

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
        self.image_size = int(math.ceil(self.image_size / stem_args.stride))

        # Build blocks
        # block 0 is the stem, block -1 is the head
        self._blocks = nn.ModuleDict({})
        for i, block_args in enumerate(self.blocks_args_list[1:-1]):
            if self.verbose > 3:
                print(f"Building block: {i+1}")
            self._blocks[f"block{i+1}"] = self.create_block(block_args)

        # Head
        out_channels = self.build_head(verbose, fixed_params)

        # pooling
        pool_size, linear_input_size = pool_like(
            out_channels=out_channels, 
            num_classes=num_classes, 
            image_size=self.image_size
        )
        self._avg_pooling = nn.AdaptiveAvgPool2d(pool_size)

        self.dropout = nn.Dropout(self.dropout_rate)   
        self.fc = nn.Linear(linear_input_size, num_classes)

        if pre_trained:
            self.load_pre_trained(pre_trained)
        
    
    def forward(self, inputs):
        x = GroupTensor(inputs, self.input_field_type)
        # Stem
        x = self._conv_stem(x)
        # Blocks
        for _, block in self._blocks.items():
            for i, restrict_or_layer in enumerate(block):
                x = restrict_or_layer(x)

        # Head
        if self.head_exists:
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
    
    def create_block(self, block_args: BlockArgs) -> nn.Module:
        layers = nn.ModuleList([])

        # restriction
        group_id = get_group_id(block_args.reflection, block_args.group)
        restrict = Restriction_Group_or_CNN(self.field_type,group_id)
        self.restrict_last = restrict
        layers.append(restrict)
        self.field_type = restrict.out_type

        # The first layer needs to take care of stride and filter size increase.
        layers.append(self.create_conv_layer(restrict, block_args))
        self.image_size = int(math.ceil(self.image_size / block_args.stride))
        
        # Add rest of layers
        block_args.stride = 1
        for _ in range(block_args.num_layers - 1):
            layers.append(
                self.create_conv_layer(restrict, block_args)
            )

        return layers


    def create_conv_layer(
        self,
        restrict: nn.Module,
        block_args: BlockArgs,
    )-> nn.Module:  
        setting = restrict.setting
        if setting in ["cnn", "switch"]:
            layer = NAS_layer(
                    in_channel_size=self.field_type,
                    block_args=block_args,
                    image_size=self.image_size,
                    dropout_rate=self.dropout_rate,
                    expand_ratio=self.cnn_expand_ratio,
                )
        else:
            layer = Eq_NAS_layer(
                    in_type=self.field_type,
                    in_channel_size=self.prev_channel_size,
                    fixed_params=self.fixed_params, 
                    block_args=block_args, 
                    image_size=self.image_size,
                    dropout_rate=self.dropout_rate,
                    expand_ratio=self.eq_expand_ratio,
                )
        self.prev_channel_size = block_args.out_channel
        self.field_type = layer.out_type
            
        return layer

    def build_head(self, verbose, fixed_params):
        last_block_args = self.blocks_args_list[-1]
        self.head_exists = last_block_args.kernel_size > 0
        if self.head_exists: 
            # Build head
            if verbose > 3:
                print("Building head")

            # Restrict
            group_id = get_group_id(last_block_args.reflection, last_block_args.group)
            self.restrict_last = Restriction_Group_or_CNN(self.field_type, group_id)  
            self.field_type = self.restrict_last.out_type
        
    
            if self.restrict_last.setting in ["cnn", "switch"]:
                self._bn1 = nn.BatchNorm2d(self.field_type)
                self._swish1 = nn.SiLU()
                self._conv_head = Conv2dSamePadding(
                    in_channels=self.field_type,
                    out_channels=last_block_args.out_channel,
                    kernel_size=last_block_args.kernel_size,
                    bias=False,
                )
            elif self.restrict_last.setting == "group": 
                out_channels = adjusted_out_channels(
                    out_channel = last_block_args.out_channel,
                    N=self.field_type.gspace.fibergroup.order(),
                    fixed_params=fixed_params,
                )
                self._bn1 = BatchNorm(in_type=self.field_type, affine=False)
                self._swish1 = Swish(in_type=self._bn1.out_type)
                self._conv_head = Eq_Conv2dSamePadding(
                    in_type=self._swish1.out_type, 
                    out_channels=out_channels,
                    bias=False
                )
                self.field_type = self._conv_head.out_type
            else:
                raise NotImplementedError(f"This setting: {self.restrict_last.setting} is not implemented")


        if self.restrict_last.setting in ["cnn", "switch"]:
            raise NotImplementedError("This setting is not implemented")
            out_channels = None
            self._bn2 = nn.BatchNorm2d(out_channels)
            self._swish2 = nn.SiLU()
        elif self.restrict_last.setting == "group": 
            self._bn2 = BatchNorm(in_type=self.field_type, affine=False)
            self._swish2 = Swish(in_type=self._bn2.out_type)
            # Final linear layer
            self.invariant_map = EquivariantPool(
                self._swish2.out_type, 
                invariant_map=True
            ) 
            out_channels = len(self.invariant_map.out_type)
        return out_channels
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        #from equivariant.nn.modules.conv import R2Conv
        print("Loading triggered for eq_nasnet")
        # check seed is the same 
        #assert self.seed == state_dict["seed"], "Save and load seeds are not the same"
        create_filters_network(self)

    def load_pre_trained(self, pre_trained: str):
        assert os.path.isfile(pre_trained), f"Pre-trained weights not found at {pre_trained}"
        state_dict = torch.load(pre_trained)
        if "state_dict" in state_dict:
            # this is a lightning checkpoint
            state_dict = state_dict["state_dict"]
            # strip the net from the keys
            state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}

        # remove the fully connected layer from the keys
        state_dict = {k: v for k, v in state_dict.items() if k not in ["fc.weight", "fc.bias"]}
        if self.num_channels != 3:
            # remove the first layer from the keys
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_conv_stem")}
        super().load_state_dict(state_dict, strict=False)

        create_filters_network(self)
        
        
