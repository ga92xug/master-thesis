import warnings
import numpy as np
from typing import Tuple
import torch
from torch import nn
import os
import sys
sys.path.append(f"{os.getcwd()}")

from equivariant.nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    SequentialModule,
    GroupTensor,
)
from networks import (
    Restriction,
    EquivariantPool,
    EquivariantNorm,
    EquivariantConvBlock,
)


class EquivariantResNet9(nn.Module):
    def __init__(
        self,
        group: str = "cyclic",  # "dihedral", "orthogonal"
        rotation: int = 4,  # discrete number or frequency
        fix_params: bool = False,
        restrict: str = None,  # "invariant", "reflection", "halved"
        num_channels: int = 3,
        layout: Tuple[int] = (64, 128, 256),
        kernel_size: int = 3,
        padding: int = 1,
        num_groups: Tuple[int] = (1,1,1), # (None, None, None),
        num_classes: int = 10,
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

        # Fix number of parameters for all groups
        num_channels = np.array(layout)
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
                    (layout[2] * np.sqrt(0.75 * gspace.fibergroup.order()))
                    / (gspace.fibergroup.order() / 2)
                )
                if num_channels[2] < 1:
                    warnings.warn(
                        f"Group order ({gspace.fibergroup.order()/2}) is larger"
                        f" than number of channels ({layout[2]}) defined in layout!"
                    )
                    num_channels[2] = 1
            elif restrict == "reflection":
                num_channels[2] = int(num_channels[2] * np.sqrt(3) / 2)
            elif restrict == "invariant":
                num_channels[2] = int(num_channels[2] * np.sqrt(1.5))

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            gspace, [gspace.trivial_repr] * num_channels
        )

        # "Lifting" conv from trivial to regular feature fields
        self.conv1 = EquivariantConvBlock(
            in_type=self.input_field_type,
            out_channels=num_channels[0],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[0],
        )

        self.conv2 = EquivariantConvBlock(
            in_type=self.conv1.out_type,
            out_channels=num_channels[1],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[1],
            pool_size=2,
        )

        res1 = [
            EquivariantConvBlock(
                in_type=self.conv2.out_type,
                out_channels=num_channels[1],
                frequency=rotation,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups[1],
            )
        ]
        res1 += [
            EquivariantConvBlock(
                in_type=res1[-1].out_type,
                out_channels=num_channels[1],
                frequency=rotation,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups[1],
            )
        ]
        self.res1 = SequentialModule(*res1)

        self.scale_norm1 = EquivariantNorm(
            in_type=self.res1.out_type, num_groups=num_groups[1], affine=False
        )

        self.conv3 = EquivariantConvBlock(
            in_type=self.scale_norm1.out_type,
            out_channels=num_channels[2],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[2],
            pool_size=2,
        )

        # Restrict last conv and res layers
        self.restrict = Restriction(self.conv3.out_type, group, rotation, restrict)

        self.conv4 = EquivariantConvBlock(
            in_type=self.restrict.out_type,
            out_channels=num_channels[2],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[2],
            pool_size=2,
        )

        res2 = [
            EquivariantConvBlock(
                in_type=self.conv4.out_type,
                out_channels=num_channels[2],
                frequency=rotation,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups[2],
            )
        ]
        res2 += [
            EquivariantConvBlock(
                in_type=res2[-1].out_type,
                out_channels=num_channels[2],
                frequency=rotation,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups[2],
            )
        ]
        self.res2 = SequentialModule(*res2)

        self.scale_norm2 = EquivariantNorm(
            in_type=self.res2.out_type, num_groups=num_groups[2], affine=False
        )

        self.invariant_map = EquivariantPool(
            in_type=self.scale_norm2.out_type, invariant_map=True
        )
        # self.global_pool = Reduce("N C (H 2) (W 2) -> N C H W", "mean")
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size * 2 * 2, num_classes
        )

    def forward(self, x):
        # Wrap input tensor in a GroupTensor
        x = GroupTensor(x, self.input_field_type)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.scale_norm1(x)
        x = self.conv3(x)
        x = self.restrict(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.scale_norm2(x)
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # H x W
    # ((H-K+2P)/S+1) x ((W-K+2P)/S+1)
    # ((12-3+2*1)/1+1) x ((12-3+2*1)/1+1) = 12 x 12
    # ((12-5+2*1)/1+1) x ((12-5+2*1)/1+1) = 10 x 10
    inp = torch.rand(1, 1, 28, 28).cuda()
    model = EquivariantResNet9(kernel_size=5, num_channels=inp.size(1), padding=2).cuda()
    # inp = torch.rand(1, 3, 32, 32).cuda()
    #inp = torch.rand(1, 3, 32, 32).cuda()
    #model = EquivariantResNet9().cuda()
    out = model(inp)
    print(out.shape)
