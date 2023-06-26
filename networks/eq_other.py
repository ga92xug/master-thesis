from typing import Tuple, List
from torch import nn
import numpy as np
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory
import nn as nn_eq

from nn import (
    FieldType,
    EquivariantModule,
    BatchNorm,
    GroupPooling,
    PointwiseMaxPool,
    MultipleModule,
)
from group_theory import Representation

__all__ = [
    "Restriction",
    "EquivariantNorm",
    "EquivariantPool",
]


def only_zero_freq(repr: Representation):
    for irr in repr.irreps:
        idx = list(
            filter(
                lambda x: repr.group.irreps()[x].name == f"irrep_{irr[0]},{irr[1]}",
                range(len(repr.group.irreps())),
            )
        )
        if repr.group.irreps()[idx[0]].attributes["frequency"] != 0:
            return False
    return True


class EquivariantNorm(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        affine: bool = False,
    ):
        super().__init__()
        self.in_type = in_type

        # batch norm
        norm = BatchNorm
        param = affine

        self.norm = norm(self.in_type, param)
        

        self.out_type = self.norm.out_type

    def forward(self, x):
        return self.norm(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantPool(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        pool_size: int = None,
        invariant_map: bool = False,
    ):
        super().__init__()
        self.in_type = in_type
        self.map = None
        self.pool = None

        labels = ["pointwise"] * len(self.in_type)
        modules_map = [(GroupPooling(self.in_type), "pointwise")]
        modules_pool = [(PointwiseMaxPool(self.in_type, pool_size), "pointwise")]
    

        if invariant_map:
            self.map = MultipleModule(self.in_type, labels, modules_map)
            if pool_size is not None:
                self.pool = PointwiseMaxPool(self.out_type, pool_size)
                self.out_type = self.pool.out_type
            else:
                self.pool = nn.Identity()
                self.out_type = self.map.out_type
        elif pool_size is not None:
            self.pool = MultipleModule(self.in_type, labels, modules_pool)
            self.out_type = self.pool.out_type
        else:
            raise ValueError(
                "EquivariantPool got no values for pooling and/or invariant mapping."
            )

    def forward(self, x):
        if self.map:
            x = self.map(x)
        x = self.pool(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

