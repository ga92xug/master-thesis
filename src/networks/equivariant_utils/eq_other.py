from typing import Tuple
from torch import nn
import os
import sys
sys.path.append(os.getcwd())

from equivariant.nn import (
    FieldType,
    EquivariantModule,
    BatchNorm,
    GroupPooling,
    PointwiseMaxPool,
    MultipleModule,
)
from equivariant.group_theory import Representation

__all__ = [
    "Restriction",
    "EquivariantNorm",
    "EquivariantPool",
]


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

