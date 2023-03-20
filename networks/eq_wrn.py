import warnings
from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Variable

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
)


class EquivariantWideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 4,
        group: str = "cyclic",
        rotation: int = None,
        fix_params: bool = False,
        restrict: str = None,  # "invariant", "reflection", "halved"
        input_channels: int = 3,
        layout: Tuple[int] = (16, 32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        num_groups: Tuple[int] = (None, None, None),
        num_classes: int = 10,
    ):
        super(EquivariantWideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n+4."
        n = (depth - 4) / 6
        k = widen_factor

        print("| Wide-Resnet %dx%d" % (depth, k))

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

        # Fix number of parameters for all groups
        num_channels = np.array(layout)
        if fix_params:  # values heuristically found
            for l in range(len(num_channels)):
                if group == "orthogonal":
                    num_channels[l] = int(num_channels[l] / (rotation + 0.9))
                else:
                    num_channels[l] = int(
                        (
                            num_channels[l]
                            * np.sqrt(1.25 * gspace.fibergroup.rotation_order)
                        )
                        / gspace.fibergroup.rotation_order
                    )
                if num_channels[l] < 1:
                    warnings.warn(
                        f"Group order ({gspace.fibergroup.rotation_order}) is larger"
                        f" than number of channels ({num_channels[l]}) defined in layout!"
                    )
                    num_channels[l] = 1
            if restrict == "halved":
                num_channels[2] = int(
                    (layout[2] * np.sqrt(0.65 * gspace.fibergroup.rotation_order))
                    / (gspace.fibergroup.rotation_order / 2)
                )
                if num_channels[2] < 1:
                    warnings.warn(
                        f"Group order ({gspace.fibergroup.rotation_order/2}) is larger"
                        f" than number of channels ({layout[2]}) defined in layout!"
                    )
                    num_channels[2] = 1
            elif restrict == "reflection":
                num_channels[2] = int(num_channels[2] * np.sqrt(3) / 2)
            elif restrict == "invariant":
                num_channels[2] = int(num_channels[2] * np.sqrt(1.5))

        # Add width
        num_channels *= np.array([k, k, k])

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            gspace, [gspace.trivial_repr] * input_channels
        )

        # "Lifting" conv from trivial to regular feature fields
        self.layer1 = EquivariantConvBlock(
            in_type=self.input_field_type,
            out_channels=int(num_channels[0] / k),
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[0],
        )

        self.field_type = self.layer1.out_type

        self.layer2 = self._wide_layer(
            EquivariantWideConvBlock,
            num_channels[0],
            n,
            stride=1,
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[0],
        )
        self.layer3 = self._wide_layer(
            EquivariantWideConvBlock,
            num_channels[1],
            n,
            stride=2,
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[1],
        )

        # Restrict last conv and res layers
        self.restrict = Restriction(self.layer3.out_type, group, rotation, restrict)
        self.field_type = self.restrict.out_type

        self.layer4 = self._wide_layer(
            EquivariantWideConvBlock,
            num_channels[2],
            n,
            stride=2,
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[2],
        )

        self.invariant_map = EquivariantPool(self.layer4.out_type, invariant_map=True)
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size * 2 * 2, num_classes
        )

    def _wide_layer(
        self,
        block,
        out_channels: int,
        num_blocks: int,
        stride: int,
        frequency: int,
        kernel_size: int,
        padding: int,
        num_groups: int,
    ):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    self.field_type,
                    out_channels,
                    stride=stride,
                    frequency=frequency,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_groups=num_groups,
                )
            )
            self.field_type = layers[-1].out_type

        return SequentialModule(*layers)

    def forward(self, x):
        # Wrap input tensor in a GroupTensor
        x = GroupTensor(x, self.input_field_type)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.restrict(x)
        x = self.layer4(x)
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = EquivariantWideResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1, 3, 32, 32)))

    print(y.size())
