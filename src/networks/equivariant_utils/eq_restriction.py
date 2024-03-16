from operator import is_
from typing import Tuple, List, Union
from torch import nn
import numpy as np
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

from equivariant.nn import (
    FieldType,
    EquivariantModule,
    SequentialModule,
    DisentangleModule,
    RestrictionModule,
)
from src.networks.equivariant_utils.eq_other import EquivariantPool

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

        old_group_id = in_type.gspace._sg_id

        if old_group_id == group_id or \
            ((group_id[0] is not None and old_group_id[0] is not None) \
             and group_id[1] == old_group_id[1]):
            # no restriction
            #print("No restriction", group_id, old_group_id)
            self.restrict = nn.Identity()
            self.out_type = self.in_type
            return
        else:
            # restriction
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


class Restriction_Group_or_CNN(nn.Module):
    """
    Restriction for the group or switch to CNN.
    """
    def __init__(
        self, 
        in_type: Union[FieldType, int], 
        group_id: Tuple,
    ):
        super().__init__()
        self.is_cnn = group_id[1] == 0

        if self.is_cnn and isinstance(in_type, FieldType):
            # switch to CNN
            self.setting = "switch"
            self.invariant_map = EquivariantPool(
                in_type, 
                invariant_map=True
            )
            self.out_type = len(in_type)
            print("Switching to CNN.")
            print("in_type", in_type)
            print("out_type", self.out_type)
        elif self.is_cnn:
            # already in CNN
            self.setting = "cnn"
            self.out_type = in_type
        else:
            # group
            self.setting = "group"
            assert isinstance(in_type, FieldType), \
            "If we are in the group setting, in_type has to be a FieldType."
            self.restrict = Restriction_from_id(in_type, group_id)
            self.out_type = self.restrict.out_type


    def forward(self, x):
        if self.setting == "switch":
            x = self.invariant_map(x)
            return x.tensor
        elif self.setting == "cnn":
            return x
        else:
            return self.restrict(x)


    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape