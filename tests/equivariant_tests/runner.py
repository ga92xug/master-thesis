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
)

from tests.equivariant_tests.modules import *


def check_layer_equivariance(
        rotations: List[int] = [1, 2, 4], 
        in_channels: int = 8,
        device: str = "cpu"
    ):
    """Runs equivariance check from EquivariantModule on pre-defined equivariant layers.
        The transformations applied during the check are defined by the transform function in FieldType.

        ** Currently only supported for discrete groups! **
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
            x = GroupTensor(
                torch.randn(16, input_field_type.size, 10, 10).to(device), input_field_type
            )
            test_one_layer_modules(input_field_type, x, device)

            test_multilayer_modules(input_field_type, x, device)

            test_special_layers(input_field_type, x, device)

            print(f"\nINFO: Equivariance check for group {gspace} successful.")


if __name__ == "__main__":
    check_layer_equivariance()
