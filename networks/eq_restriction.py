from operator import is_
from typing import Tuple, List
from torch import nn
import numpy as np
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

from nn import (
    FieldType,
    EquivariantModule,
    SequentialModule,
    DisentangleModule,
    RestrictionModule,
)
from networks.eq_other import EquivariantPool

class Restriction(EquivariantModule):
    def __init__(
        self, in_type: FieldType, group: str, rotation: int, restrict: str = None
    ):
        super().__init__()
        self.in_type = in_type
        if restrict == "none" or restrict is None:
            self.restrict = nn.Identity()
            self.out_type = self.in_type
        else:
            layers = list()

            if restrict == "reflection":
                assert group != "cyclic", "Cyclic groups can't be restricted to reflection."
                subgroup_id = (np.pi, 1) if group == "orthogonal" else (0, 1)

            elif restrict == "halved":
                assert group != "orthogonal", "Orthogonal group can't be restricted by halve."
                assert rotation % 2 == 0, f"Number of rotations ({rotation}) is not divisible by 2."
                subgroup_id = (0, rotation // 2) if group == "dihedral" else (rotation // 2)
                
            elif restrict == "invariant":  
                # restrict to invariant case
                subgroup_id = (None, 1) if group != "cyclic" else 1
            else:
                raise ValueError(f"Restriction {restrict} not implemented.")

            layers.append(RestrictionModule(self.in_type, subgroup_id))
            layers.append(DisentangleModule(layers[-1].out_type))
            self.restrict = SequentialModule(*layers)
            self.out_type = self.restrict.out_type

    def forward(self, x):
        return self.restrict(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class Restriction_from_id(EquivariantModule):
    """
    If we restrict the cyclic group further the group_id has to be rotations
    else it is a tuple of (reflection, rotation).
    """
    def __init__(
        self, in_type: FieldType, group_id: Tuple,
    ):
        super().__init__()
        self.in_type = in_type
        self.restriction_correction_factor = 1

        if "C" in self.in_type.fibergroup.name:
            # cyclic group
            self.restrict = RestrictionModule(self.in_type, group_id[1])
        elif "D" in self.in_type.fibergroup.name:
            # dihedral group
            self.restrict = RestrictionModule(self.in_type, group_id)
            if group_id[0] is None:
                # if we change from dihedral to cyclic group
                self.restriction_correction_factor *= 2
        else:  
            ValueError("Only cyclic and dihedral groups are supported.")
        
        old_rotation = self.in_type.gspace._sg_id[1]
        self.restriction_correction_factor *= old_rotation / group_id[1]
        self.out_type = self.restrict.out_type

    def forward(self, x):
        return self.restrict(x)

    def get_correction_factor(self):
        #print(self.restriction_correction_factor)
        return self.restriction_correction_factor

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class Restriction_Group_or_CNN():
    def __init__(
        self, in_type: FieldType, group_id: Tuple,
    ):
        self.is_cnn = group_id[0] == 0

        if self.is_cnn:
            # CNN
            pass
            self.invariant_map = EquivariantPool(
                in_type, 
                invariant_map=True
            )
            channels = len(in_type)
        else:
            # group
            self.restrict = Restriction_from_id(in_type, group_id)
            self.out_type = self.restrict.out_type
        

    def forward(self, x):
        if self.is_cnn:
            pass
        else:
            pass

        return self.restrict(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape