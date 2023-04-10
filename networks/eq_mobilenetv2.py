import warnings
from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

import numpy as np

from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    SequentialModule,
    GroupTensor,
)
from networks import (
    Restriction,
    EquivariantPool,
    EquivariantConvBlock,
    EquivariantWideConvBlock,
    EquivariantConvBlock_Conv_BN_actF,
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
        fix_params: bool = False,
        restrict: str = None,  # "invariant", "reflection", "halved"
        input_channels: int = 3,
        channel_layout: Tuple[int] = (32, 16, 24, 32, 64, 96, 160, 320, 1280),
        bottleneck_layout: Tuple[int] = (1, 2, 3, 4, 3, 3, 1),
        kernel_size: int = 3,
        padding: int = 1,
        num_groups: Tuple[int] = (None, None, None, None, None, None,\
                                  None, None, None), # (None x 9) for no group equivariance
        num_classes: int = 10,
        expand_ratio: int = 6,
        size_reduction: int = 1,

    ):
        super().__init__()
        # Get group spaces for specified rotations and flips
        if group == "cyclic":
            gspace = rot2dOnR2(rotation)
        elif group == "dihedral":
            gspace = flipRot2dOnR2(rotation)
        elif group == "orthogonal":
            gspace = flipRot2dOnR2(-1)
        else:
            raise ValueError(
                f'Group "{group}" is not know. Available groups: [cyclic, dihedral, orthogonal]'
            )

        # Normalize channel layout by group order
        num_channels = (np.array(channel_layout) / ((gspace.fibergroup.order()) * size_reduction)).astype(int)
        # print(num_channels)
        # Fix number of parameters for all groups
        # these values are not yet correct. Taken from ResNet9
        if fix_params:  # values heuristically found
            for l in range(len(num_channels)):
                if group == "orthogonal":
                    num_channels[l] = int(num_channels[l] / np.sqrt(1.75 * rotation))
                else:
                    num_channels[l] = int(
                        (num_channels[l] * np.sqrt(1.5 * gspace.fibergroup.order()))
                        / gspace.fibergroup.order()
                    )
                if num_channels[l] < 1:
                    warnings.warn(
                        f"Group order (number of rotations or frequency) is larger"
                        f" than number of channels ({num_channels[l]}) defined in layout!"
                    )
                    num_channels[l] = 1
            if restrict == "halved":
                num_channels[2] = int(
                    (channel_layout[2] * np.sqrt(0.75 * gspace.fibergroup.order()))
                    / (gspace.fibergroup.order() / 2)
                )
                if num_channels[2] < 1:
                    warnings.warn(
                        f"Group order ({gspace.fibergroup.order()/2}) is larger"
                        f" than number of channels ({channel_layout[2]}) defined in layout!"
                    )
                    num_channels[2] = 1
            elif restrict == "reflection":
                num_channels[2] = int(num_channels[2] * np.sqrt(3) / 2)
            elif restrict == "invariant":
                num_channels[2] = int(num_channels[2] * np.sqrt(1.5))

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            gspace, [gspace.trivial_repr] * input_channels
        )

        # conv 1
        self.conv1 = EquivariantConvBlock_Conv_BN_actF(
            in_type=self.input_field_type,
            out_channels=num_channels[0],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[0],
            act_func="ReLU",
            stride=2,
        )
        # block 0
        self.bottleneck_block_0 = EquivariantBottleneckBlock(
            in_type=self.conv1.out_type,
            out_channels=num_channels[1],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            num_groups=num_groups[1],
            act_func="ReLU",
            expand_ratio=1,
            num_blocks=bottleneck_layout[0],
        )
        # block 1
        self.bottleneck_block_1 = EquivariantBottleneckBlock(
            in_type=self.bottleneck_block_0.out_type,
            out_channels=num_channels[2],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
            num_groups=num_groups[2],
            act_func="ReLU",
            expand_ratio=expand_ratio,
            num_blocks=bottleneck_layout[1],
        )
        # block 2
        self.bottleneck_block_2 = EquivariantBottleneckBlock(
            in_type=self.bottleneck_block_1.out_type,
            out_channels=num_channels[3],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
            num_groups=num_groups[3],
            act_func="ReLU",
            expand_ratio=expand_ratio,
            num_blocks=bottleneck_layout[2],
        )
        # block 3
        self.bottleneck_block_3 = EquivariantBottleneckBlock(
            in_type=self.bottleneck_block_2.out_type,
            out_channels=num_channels[4],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
            num_groups=num_groups[4],
            act_func="ReLU",
            expand_ratio=expand_ratio,
            num_blocks=bottleneck_layout[3],
        )
        # block 4
        self.bottleneck_block_4 = EquivariantBottleneckBlock(
            in_type=self.bottleneck_block_3.out_type,
            out_channels=num_channels[5],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            num_groups=num_groups[5],
            act_func="ReLU",
            expand_ratio=expand_ratio,
            num_blocks=bottleneck_layout[4],
        )
        # block 5
        self.bottleneck_block_5 = EquivariantBottleneckBlock(
            in_type=self.bottleneck_block_4.out_type,
            out_channels=num_channels[6],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
            num_groups=num_groups[6],
            act_func="ReLU",
            expand_ratio=expand_ratio,
            num_blocks=bottleneck_layout[5],
        )
        # block 6
        self.bottleneck_block_6 = EquivariantBottleneckBlock(
            in_type=self.bottleneck_block_5.out_type,
            out_channels=num_channels[7],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            num_groups=num_groups[7],
            act_func="ReLU",
            expand_ratio=expand_ratio,
            num_blocks=bottleneck_layout[6],
        )
        # conv 2
        self.conv2 = EquivariantConvBlock_Conv_BN_actF(
            in_type=self.bottleneck_block_6.out_type,
            out_channels=num_channels[8],
            frequency=rotation,
            kernel_size=1,
            padding=0,
            num_groups=num_groups[8],
        )
        self.invariant_map = EquivariantPool(self.conv2.out_type, invariant_map=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size, num_classes
        )

        #self.avgpool = GeometricAveragePooling(self.conv2.out_type)
        #self.fc = GeometricLinear(self.avgpool.out_type, num_classes)

    def forward(self, x):
        x = GroupTensor(x, self.input_field_type)
        x = self.conv1(x)
        x = self.bottleneck_block_0(x)
        x = self.bottleneck_block_1(x)
        x = self.bottleneck_block_2(x)
        x = self.bottleneck_block_3(x)
        x = self.bottleneck_block_4(x)
        x = self.bottleneck_block_5(x)
        x = self.bottleneck_block_6(x)
        x = self.conv2(x)
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    # input images from the paper 224 x 224 x 3
    inp = torch.rand(1, 3, 224, 224).cuda()
    model = EquivariantMobileNetV2().cuda()
    out = model(inp)
    print(out.shape)