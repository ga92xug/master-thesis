import numpy as np
from typing import List, Tuple
from omegaconf import DictConfig
import torch.nn as nn
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

from equivariant.nn import (
    FieldType,
    GroupTensor,
)
from networks.util import (
    calculate_output_image_size,
    get_fixed_params, 
    get_gspace_from_name, 
)
from networks import (
    Restriction,
    EquivariantPool,
    EquivariantConvBlock,
    Equivariant_Conv_BN_actF,
)

from .util import (
    EquivariantBottleneck,
    EquivariantBottleneckBlock,
)

# adapted from https://github.com/mnmjh1215/mobilenet-pytorch/blob/master/models/mobilenetv2.py
# building blocks for mobilenet-v2. (https://arxiv.org/pdf/1801.04381.pdf)

class EquivariantMobileNetV2(nn.Module):
    def __init__(
        self,
        group: str = "cyclic",  # "dihedral", "orthogonal"
        rotation: int = 4,  # discrete number or frequency
        fix_params_mode: str = "heuristic",
        restrict: List[str] = [None] * 7,  # "invariant", "reflection", "halved"
        input_channels: int = 3,
        channel_layout: List[int] = [32, 16, 24, 32, 64, 96, 160, 320, 1280],
        bottleneck_layout: List[int] = [1, 2, 3, 4, 3, 3, 1],
        kernel_size: int = 3,
        padding: int = 1,
        num_classes: int = 10,
        expand_ratio: int = 6,
        image_size: int = 32,
        depth_multiplier: int = 1,
        width_multiplier: int = 1,
        min_feature_map_size: int = 5,
        drop_out: float = 0.0,
    ):
        super().__init__()
        print(f"Eq_MobileNetV2_{depth_multiplier}_{width_multiplier}_{group}_{rotation}")
        self.drop_out = drop_out
        self.restrict = list(restrict)
        gspace = get_gspace_from_name(group, rotation)

        self.num_channels = np.round((np.array(channel_layout) * width_multiplier))
        self.bottleneck_layout = (np.round(np.array(bottleneck_layout) * depth_multiplier)).astype(int) 
        
        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(gspace, [gspace.trivial_repr] * input_channels)

        # conv 1
        kwargs = {"in_type": self.input_field_type, "out_channels": self.num_channels[0],
            "kernel_size": kernel_size, "padding": padding, "act_func": "ReLU",
            "stride": 2}
        self.conv1 = get_fixed_params(Equivariant_Conv_BN_actF, fix_params_mode, 
                        gspace=self.input_field_type.gspace, **kwargs)
        image_size = calculate_output_image_size(image_size, stride=2) # 16

        # block 0
        kwargs = {
            "in_type": self.conv1.out_type,
            "out_channels": self.num_channels[1],
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": 1,
            "act_func": "ReLU",
            "expand_ratio": 1,
            "num_blocks": self.bottleneck_layout[0],
        }
        self.bottleneck_block_0 = get_fixed_params(EquivariantBottleneckBlock, 
                                fix_params_mode,
                                gspace=self.conv1.out_type.gspace, **kwargs)
        self.restrict_0 = Restriction(self.bottleneck_block_0.out_type, group, 
                                      rotation, self.restrict[-7])
        image_size = calculate_output_image_size(image_size, stride=1) # 16
        # block 1
        kwargs = {
            "in_type": self.restrict_0.out_type,
            "out_channels": self.num_channels[2],
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": 2,
            "act_func": "ReLU",
            "expand_ratio": expand_ratio,
            "num_blocks": self.bottleneck_layout[1],
        }
        self.bottleneck_block_1 = get_fixed_params(EquivariantBottleneckBlock, 
                        fix_params_mode,
                        gspace=self.bottleneck_block_0.out_type.gspace,
                        **kwargs)
        self.restrict_1 = Restriction(self.bottleneck_block_1.out_type, group, 
                                      rotation, self.restrict[-6])
        image_size = calculate_output_image_size(image_size, stride=2) # 8

        # block 2
        kwargs = {
            "in_type": self.restrict_1.out_type,
            "out_channels": self.num_channels[3],
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": 2,
            "act_func": "ReLU",
            "expand_ratio": expand_ratio,
            "num_blocks": self.bottleneck_layout[2],
        }
        self.bottleneck_block_2 = get_fixed_params(EquivariantBottleneckBlock, 
                        fix_params_mode,
                        gspace=self.bottleneck_block_1.out_type.gspace, **kwargs)
        self.restrict_2 = Restriction(self.bottleneck_block_2.out_type, group, rotation, self.restrict[-5])
        image_size = calculate_output_image_size(image_size, stride=2) # 4
        if image_size[0] < min_feature_map_size:
            kwargs = {
                "in_type": self.restrict_2.out_type,
                "out_channels": self.num_channels[4],
                "kernel_size": 1,
                "padding": 0,
                "stride": 1,
                "act_func": "ReLU",
            }
            self.conv2 = get_fixed_params(EquivariantConvBlock, fix_params_mode,
                            gspace=self.restrict_2.out_type.gspace, **kwargs)
            self.invariant_map = EquivariantPool(self.conv2.out_type, invariant_map=True)
            image_size = max(int(image_size[0] / 2), 1)
            self.flatten = nn.Flatten()
            self.classifier = nn.Linear(
                self.invariant_map.out_type.size  * image_size * image_size, num_classes
            )
            return

        # block 3
        kwargs = {
            "in_type": self.restrict_2.out_type,
            "out_channels": self.num_channels[4],
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": 2,
            "act_func": "ReLU",
            "expand_ratio": expand_ratio,
            "num_blocks": self.bottleneck_layout[3],
        }
        self.bottleneck_block_3 = get_fixed_params(EquivariantBottleneckBlock, fix_params_mode,
                        gspace=self.bottleneck_block_2.out_type.gspace, **kwargs)
        self.restrict_3 = Restriction(self.bottleneck_block_3.out_type, group, rotation, self.restrict[-4])
        image_size = calculate_output_image_size(image_size, stride=2) # 2
        # block 4
        kwargs = {
            "in_type": self.restrict_3.out_type,
            "out_channels": self.num_channels[5],
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": 1,
            "act_func": "ReLU",
            "expand_ratio": expand_ratio,
            "num_blocks": self.bottleneck_layout[4],
        }
        self.bottleneck_block_4 = get_fixed_params(EquivariantBottleneckBlock, fix_params_mode,
                        gspace=self.bottleneck_block_3.out_type.gspace, **kwargs)
        self.restrict_4 = Restriction(self.bottleneck_block_4.out_type, group, rotation, self.restrict[-3])
        image_size = calculate_output_image_size(image_size, stride=1) # 2
        # block 5
        kwargs = {
            "in_type": self.restrict_4.out_type,
            "out_channels": self.num_channels[6],
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": 2,
            "act_func": "ReLU",
            "expand_ratio": expand_ratio,
            "num_blocks": self.bottleneck_layout[5],
        }
        self.bottleneck_block_5 = get_fixed_params(EquivariantBottleneckBlock, fix_params_mode,
                        gspace=self.restrict_4.out_type.gspace, **kwargs)
        self.restrict_5 = Restriction(self.bottleneck_block_5.out_type, group, rotation, self.restrict[-2])
        image_size = calculate_output_image_size(image_size, stride=2) # 1
        # block 6
        kwargs = {
            "in_type": self.restrict_5.out_type,
            "out_channels": self.num_channels[7],
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": 1,
            "act_func": "ReLU",
            "expand_ratio": expand_ratio,
            "num_blocks": self.bottleneck_layout[6],
        }
        self.bottleneck_block_6 = get_fixed_params(EquivariantBottleneckBlock, fix_params_mode,
                        gspace=self.restrict_5.out_type.gspace, **kwargs)
        self.restrict_6 = Restriction(self.bottleneck_block_6.out_type, group, rotation, self.restrict[-1])
        image_size = calculate_output_image_size(image_size, stride=1)
        # conv 2
        kwargs = {
            "in_type": self.restrict_6.out_type,
            "out_channels": self.num_channels[8],
            "kernel_size": 1,
            "padding": 0,
            "stride": 1,
            "act_func": "ReLU",
        }
        self.conv2 = get_fixed_params(EquivariantConvBlock, fix_params_mode,
                        gspace=self.restrict_6.out_type.gspace, **kwargs)
        
        self.invariant_map = EquivariantPool(self.conv2.out_type, invariant_map=True)
        image_size = max(int(image_size[0] / 2), 1)
        #self.global_pool = nn.AdaptiveAvgPool2d(2) if image_size > 1 else nn.Identity()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size  * image_size * image_size, num_classes
        )

        #self.avgpool = GeometricAveragePooling(self.conv2.out_type)
        #self.fc = GeometricLinear(self.avgpool.out_type, num_classes)

    def forward(self, x):
        x = GroupTensor(x, self.input_field_type)
        x = self.conv1(x)
        x = self.bottleneck_block_0(x)
        x = self.restrict_0(x)
        x = self.bottleneck_block_1(x)
        x = self.restrict_1(x)
        x = self.bottleneck_block_2(x)
        x = self.restrict_2(x)
        x = self.bottleneck_block_3(x) if hasattr(self, "bottleneck_block_3") else x
        x = self.restrict_3(x) if hasattr(self, "restrict_3") else x
        x = self.bottleneck_block_4(x) if hasattr(self, "bottleneck_block_4") else x
        x = self.restrict_4(x) if hasattr(self, "restrict_4") else x
        x = self.bottleneck_block_5(x) if hasattr(self, "bottleneck_block_5") else x
        x = self.restrict_5(x) if hasattr(self, "restrict_5") else x
        x = self.bottleneck_block_6(x) if hasattr(self, "bottleneck_block_6") else x
        x = self.restrict_6(x) if hasattr(self, "restrict_6") else x
        x = self.conv2(x) 
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = nn.functional.avg_pool2d(x, 2) if x.shape[-1] > 1 else x
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    