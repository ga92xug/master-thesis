import sys
import os

sys.path.append(f"{os.getcwd()}")
import torch
from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    GroupTensor,
    R2Conv,
    GroupNorm,
    Mish,
    GroupPooling,
)
from networks import (
    EquivariantConvBlock,
    EquivariantNorm,
    EquivariantPool,
)

from networks.eq_convs import (
    Eq_Conv2dSamePaddingChangeFactor,
    EquivariantSqueezeExcitation,
    Equivariant_Conv_BN_actF,
)


def check_layer_equivariance(rotations: list = [1, 2, 4], in_channels: int = 8):
    """Runs equivariance check from EquivariantModule on pre-defined equivariant layers.
        The transformations applied during the check are defined by the transform function in FieldType.

        ** Currently only supported for discrete groups! **

    Current layers evaluated:
        - R2Conv
        - GroupNorm
        - GroupPooling
        - Mish

        - EquivariantConvBlock
        - EquivariantNorm
        - EquivariantPool

        - Equivariant_Conv_BN_actF
        - EquivariantSqueezeExcitation
        - Eq_Conv2dSamePaddingChangeFactor
    """
    # Iterate over different number of rotations
    for rot in rotations:
        # Cyclic and dihedral groups
        gspaces = [rot2dOnR2(rot), flipRot2dOnR2(rot)]

        for gspace in gspaces:
            print(f"\nNEXT: Group {gspace.fibergroup}")
            input_field_type = FieldType(gspace, [gspace.regular_repr] * in_channels)

            conv = R2Conv(
                input_field_type,
                input_field_type,
                kernel_size=3,
                padding=1,
                groups=1,
                stride=1,
                dilation=1,
                bias=True,
                frequencies_cutoff=lambda r: 3 * r,
            ).cuda()
            print("\nR2Conv:")
            conv.check_equivariance(atol=0.000009, rtol=0.00009)

            pool = GroupPooling(input_field_type)
            print("\nGroupPooling:")
            pool.check_equivariance()

            x = GroupTensor(
                torch.randn(16, input_field_type.size, 10, 10).cuda(), input_field_type
            )

            norm = GroupNorm(input_field_type, num_groups=3)
            print("\nGroupNorm:")
            norm.check_equivariance(x)

            act_func = Mish(input_field_type)
            print("\nMish:")
            act_func.check_equivariance(x)

            ######################################################################

            conv_block = EquivariantConvBlock(
                in_type=input_field_type,
                out_channels=16,
                #frequency=rot,
                kernel_size=3,
                padding=1,
                #num_groups=4,
            ).cuda()
            print("\nConv Block:")
            conv_block.check_equivariance()

            conv_bn_act = Equivariant_Conv_BN_actF(
                in_type=input_field_type,
                out_channels=16,
            ).cuda()
            print("\nConv BN Act:")
            conv_bn_act.check_equivariance()

            squeeze_excitation = EquivariantSqueezeExcitation(
                in_type=input_field_type,
                sequeeze_ratio=0.25,
            ).cuda()
            print("\nSqueeze Excitation:")
            squeeze_excitation.check_equivariance()

            conv_same = Eq_Conv2dSamePaddingChangeFactor(
                in_type=input_field_type,
                change_factor=1.25,
            ).cuda()
            print("\nConv Same:")
            conv_same.check_equivariance()

            #norm_block = EquivariantNorm(input_field_type, num_groups=4, affine=False)
            norm_block = EquivariantNorm(input_field_type, affine=False)
            print("\nNorm Block:")
            norm_block.check_equivariance()

            pool_block = EquivariantPool(input_field_type, invariant_map=True)
            print("\nPool Block:")
            pool_block.check_equivariance()

            print(f"\nINFO: Equivariance check for group {gspace} successful.")


if __name__ == "__main__":
    check_layer_equivariance()
