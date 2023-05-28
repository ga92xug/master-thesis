from typing import Tuple, List
from torch import nn
import numpy as np
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory
import nn as nn_eq

from nn import (
    FieldType,
    EquivariantModule,
    SequentialModule,
    BatchNorm,
    InducedNormBatchNorm,
    GroupPooling,
    NormPool,
    InducedNormPool,
    NormMaxPool,
    PointwiseMaxPool,
    DisentangleModule,
    RestrictionModule,
    MultipleModule,
)
from group_theory import Representation
from nn.modules import nonlinearities

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
        induced_norm = InducedNormBatchNorm
        param = affine

        # Split into pointwise and induced representations
        if len(set(self.in_type.fields_names())) == 1:
            self.norm = norm(self.in_type, param)
        else:
            labels = [
                "pointwise" if only_zero_freq(r) else "induced" for r in self.in_type
            ]
            field_type = self.in_type.group_by_labels(labels)
            modules = [
                (norm(field_type["pointwise"], param), "pointwise"),
                (induced_norm(field_type["induced"], param), "induced"),
            ]
            self.norm = MultipleModule(self.in_type, labels, modules)

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

        # Split into pointwise and induced representations
        if len(set(self.in_type.fields_names())) == 1:
            labels = ["pointwise"] * len(self.in_type)
            modules_map = [(GroupPooling(self.in_type), "pointwise")]
            modules_pool = [(PointwiseMaxPool(self.in_type, pool_size), "pointwise")]
        # Induced
        else:
            labels = [
                "pointwise" if only_zero_freq(r) else "induced" for r in self.in_type
            ]
            out_type = self.in_type.group_by_labels(labels)
            modules_map = [
                (NormPool(out_type["pointwise"]), "pointwise"),
                (InducedNormPool(out_type["induced"]), "induced"),
            ]
            modules_pool = [
                (PointwiseMaxPool(out_type["pointwise"], pool_size), "pointwise"),
                (NormMaxPool(out_type["induced"], pool_size), "induced"),
            ]

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


if __name__ == "__main__":
    # 
    input_channels = 3
    out_channels = 3
    r2_act = nn_eq.rot2dOnR2(N=4)
    r2_in_type = FieldType(r2_act, [r2_act.trivial_repr]*input_channels)
    
    eq_conv = EquivariantConv(in_type=r2_in_type, out_channels=out_channels, \
                              kernel_size=3, padding=1, stride=1, bias=False)
    

    conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)

    tot_param = sum([p.numel() for p in eq_conv.parameters() if p.requires_grad])
    print('Total number of parameters: {}'.format(tot_param))
    tot_param = sum([p.numel() for p in conv1.parameters() if p.requires_grad])
    print('Total number of parameters: {}'.format(tot_param))