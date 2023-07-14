from typing import Tuple
import torch
import sys
import os

sys.path.append(f"{os.getcwd()}")
from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    GroupTensor,
    FieldType,
    EquivariantModule,
    R2Conv,
    NormNonLinearity,
    Swish,
    PointwiseAdaptiveAvgPool,
    RestrictionModule,
)

from networks.util import get_group_id, get_gspace_from_id
from networks.eq_restriction import Restriction_from_id
from networks.eq_convs import EquivariantConv

from networks.gpool_reduction import GroupPoolingReduction


class TestNet(EquivariantModule):
    def __init__(self, input_channels: int = 5, reflection: int = -1, rotation: int = 4, ref_reduction: int = -1, rot_reduction: int = 4):
        super().__init__()
        group_id = get_group_id(reflection, rotation)
        gspace = get_gspace_from_id(group_id)

        restriction_id = get_group_id(ref_reduction, rot_reduction)
        #print(f"restriction id: {restriction_id}")
        if group_id[0] is None:
            restriction_id = restriction_id[1]

        self.input_field = FieldType(
                    gspace, [gspace.regular_repr] * input_channels
                )
        #print(f"field init: {self.input_field}")
        self.restriction = RestrictionModule(self.input_field, restriction_id)
        #self.restriction = Restriction_from_id(self.input_field, int(rotation/2))
        #print(f"field after restriction: {self.restriction.out_type}")
        self.conv = EquivariantConv(self.restriction.out_type, 4, kernel_size=3, padding=1)
        #print(f"field after conv: {self.conv.out_type}\n")
        self.gpool = GroupPoolingReduction(self.restriction.out_type)
        

    def forward(self, input):
        input = GroupTensor(input, self.input_field)
        out_restrict = self.restriction(input)
        out = self.conv(out_restrict)
        out_gpool = self.gpool(out_restrict)
        #print("out ", out.type, " out_gpool ", out_gpool.type)
        return out + out_gpool
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.original_in_type.size
        return input_shape


def main():
    input_channels = 4
    rotations = [2,4,8,16]
    for rot in rotations:
        # Cyclic and dihedral groups
        for reflection in [-1, 0]:
            for rot_reduction in [2,4,8,16]:
                if rot_reduction > rot:
                    continue

                for ref_reduction in [-1, 0]:
                    if ref_reduction > reflection:
                        continue
                    
                    print(f"Test with: Initial Group({reflection},{rot}), Reduction Group({ref_reduction},{rot_reduction})")

                    net = TestNet(input_channels, reflection, rot, ref_reduction, rot_reduction).cuda()
                    increase_factor = 2 if reflection == 0 else 1
                    x = torch.randn(1, input_channels * rot * increase_factor, 5, 5).cuda()
                    net(x)

                    print("Test passed")

    print("All Tests passed")

if __name__ == "__main__":
    main()