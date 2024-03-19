import sys
import os
from typing import List

from src.networks.eq_resnet.util import EquivariantWideConvBlock


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
    GSpace,
)

from src.networks.equivariant_utils.eq_convs import (
    EquivariantConvBlock,
    Eq_Conv2dSamePaddingChangeFactor,
    EquivariantSqueezeExcitation,
    Equivariant_Conv_BN_actF,
)

from src.networks.equivariant_utils.eq_other import EquivariantNorm, EquivariantPool

def test_one_layer_modules(
        input_field_type: FieldType,
        x: GroupTensor,
        device: str,
    ) -> None:
    """
    Test the basic building blocks of equivariant networks.
    """
    
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

    norm = GroupNorm(input_field_type, num_groups=3)
    print("\nGroupNorm:")
    norm.check_equivariance(x)

    act_func = Mish(input_field_type)
    print("\nMish:")
    act_func.check_equivariance(x)


def test_multilayer_modules(
        input_field_type: FieldType,
        x: GroupTensor,
        device: str,
    ) -> None:
    """
    Test equivariance of multi-layer modules commonly used to build equivariant networks.
    """

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


def test_special_layers(
        input_field_type: FieldType,
        x: GroupTensor,
        device: str,
    ) -> None:
    """
    Test equivariance of special layers built for individual use-cases.
    """    
    conv_block = EquivariantWideConvBlock(
        in_type=input_field_type,
        out_channels=16,
        #frequency=rot,
        kernel_layout=[3,1,3],
        padding=1,
        #num_groups=4,
    ).to(device)
    print("\nConv Block:")
    conv_block.check_equivariance()