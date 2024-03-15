import sys
import os


sys.path.append(f"{os.getcwd()}")
import torch
from equivariant.nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    GroupTensor,
    PointwiseDropout,
)

from src.networks.eq_convs import EquivariantConvDropOut

from src.networks.eq_resnet.util import (
    EquivariantWideConvBlock,
    EquivariantWideConvBlock_drop_out,
)


def check_layer_equivariance(rotations: list = [1, 2, 4], in_channels: int = 8):
    """Checks equivariance of Eq_WRN support modules.

    Current layers evaluated:
        - EquivariantWideConvBlock
    """
    # Iterate over different number of rotations
    for rot in rotations:
        # Cyclic and dihedral groups
        gspaces = [rot2dOnR2(rot), flipRot2dOnR2(rot)]

        for gspace in gspaces:
            print(f"\nNEXT: Group {gspace.fibergroup}")
            input_field_type = FieldType(gspace, [gspace.regular_repr] * in_channels)


            conv_block = EquivariantWideConvBlock(
                in_type=input_field_type,
                out_channels=16,
                #frequency=rot,
                kernel_layout=[3,1,3],
                padding=1,
                #num_groups=4,
            ).cuda()
            print("\nConv Block:")
            conv_block.check_equivariance()

            print(f"\nINFO: Equivariance check for group {gspace} successful.")


if __name__ == "__main__":
    check_layer_equivariance()
