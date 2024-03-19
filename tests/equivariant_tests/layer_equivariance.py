import sys
import os
from typing import List

sys.path.append(f"{os.getcwd()}")
import torch
from equivariant.nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    GroupTensor,
    R2Conv,
    GroupNorm,
    Mish,
    GroupPooling,
)
from src.networks import (
    EquivariantConvBlock,
    EquivariantNorm,
    EquivariantPool,
)

from src.networks.equivariant_utils.eq_convs import (
    Eq_Conv2dSamePaddingChangeFactor,
    EquivariantSqueezeExcitation,
    Equivariant_Conv_BN_actF,
)


def check_layer_equivariance(
        rotations: List[int] = [1, 2, 4], 
        in_channels: int = 8,
        device: str = "cpu"
    ):
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
    assert device in ["cuda", "cpu"], f"Device {device} not supported."
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available."

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
            ).to(device)
            print("\nR2Conv:")
            conv.check_equivariance(atol=0.00001, rtol=0.0001)

            pool = GroupPooling(input_field_type)
            print("\nGroupPooling:")
            pool.check_equivariance()

            x = GroupTensor(
                torch.randn(16, input_field_type.size, 10, 10).to(device), input_field_type
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
            ).to(device)
            print("\nConv Block:")
            conv_block.check_equivariance()

            conv_bn_act = Equivariant_Conv_BN_actF(
                in_type=input_field_type,
                out_channels=16,
            ).to(device)
            print("\nConv BN Act:")
            conv_bn_act.check_equivariance()

            squeeze_excitation = EquivariantSqueezeExcitation(
                in_type=input_field_type,
                sequeeze_ratio=0.25,
            ).to(device)
            print("\nSqueeze Excitation:")
            squeeze_excitation.check_equivariance()

            conv_same = Eq_Conv2dSamePaddingChangeFactor(
                in_type=input_field_type,
                change_factor=1.25,
            ).to(device)
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
