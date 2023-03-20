from typing import Tuple
from torch import nn
import numpy as np

from nn import (
    GroupTensor,
    FieldType,
    EquivariantModule,
    SequentialModule,
    R2Conv,
    GroupNorm,
    InducedNormGroupNorm,
    GroupStandardization,
    BatchNorm,
    InducedNormBatchNorm,
    Mish,
    NormNonLinearity,
    InducedGatedNonLinearity,
    GroupPooling,
    NormPool,
    InducedNormPool,
    NormAvgPool,
    NormMaxPool,
    PointwiseAvgPool,
    PointwiseAdaptiveAvgPool,
    PointwiseMaxPool,
    DisentangleModule,
    RestrictionModule,
    MultipleModule,
)
from group_theory import Representation

__all__ = [
    "Restriction",
    "EquivariantConv",
    "EquivariantNorm",
    "EquivariantPool",
    "EquivariantConvBlock",
    "EquivariantWideConvBlock",
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


class Restriction(EquivariantModule):
    def __init__(
        self, in_type: FieldType, group: str, rotation: int, restrict: str = None
    ):
        super().__init__()
        self.in_type = in_type

        if not restrict:
            self.restrict = nn.Identity()
            self.out_type = self.in_type
        else:
            layers = list()

            if restrict == "reflection":
                assert (
                    group != "cyclic"
                ), "Cyclic groups can't be restricted to reflection."

                subgroup_id = (np.pi, 1) if group == "orthogonal" else (0, 1)
            elif restrict == "halved":
                assert (
                    group != "orthogonal"
                ), "Orthogonal group can't be restricted by halve."
                assert (
                    rotation % 2 == 0
                ), f"Number of rotations ({rotation}) is not divisible by 2."

                subgroup_id = (
                    (0, rotation // 2) if group == "dihedral" else (rotation // 2)
                )
            else:  # restrict to invariant case
                subgroup_id = (None, 1) if group != "cyclic" else 1

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


class EquivariantConv(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        frequency: int = None,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        no_gates: bool = False,
    ):
        super().__init__()
        self.in_type = in_type  # declaration required by base class

        # Induced
        if self.in_type.gspace.fibergroup.name == "O(2)":
            subgroup_id = (None, self.in_type.gspace.fibergroup.rotation_order)
            sg, _, _ = self.in_type.gspace.restrict(subgroup_id)

            induced_trivial = self.in_type.gspace.induced_repr(
                subgroup_id, sg.trivial_repr
            )
            induced_irreps = []
            irrep_list = sg.fibergroup.irreps()[: frequency + 1]
            for irr in irrep_list:
                if not irr.is_trivial():
                    induced_irreps += [
                        self.in_type.gspace.induced_repr(subgroup_id, irr)
                    ]

            # Total channels = C * S   +  1 * C * len(induced_irreps)  = C * M
            # i.e.:           (fields) + (gates for non-trivial fields)

            self.trivials = FieldType(
                self.in_type.gspace, [induced_trivial] * out_channels
            )
            gates = FieldType(
                self.in_type.gspace,
                [induced_trivial] * out_channels * len(induced_irreps),
            )
            gated = FieldType(
                self.in_type.gspace, induced_irreps * out_channels
            ).sorted()
            self.gate = gates + gated

            if no_gates:  # for shortcut conv without activation
                out_type = self.trivials + gated
            else:
                out_type = self.trivials + self.gate

            self.conv = R2Conv(
                self.in_type,
                out_type,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                stride=stride,
                dilation=dilation,
                bias=bias,
                frequencies_cutoff=lambda r: 3 * r,
            )
        # Cyclic and Dihedral Groups
        else:
            out_type = FieldType(
                self.in_type.gspace,
                [self.in_type.gspace.regular_repr] * out_channels,
            )

            self.conv = R2Conv(
                self.in_type,
                out_type,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=bias,
                sigma=None,
                frequencies_cutoff=lambda r: 3 * r,
            )
            self.trivials, self.gate = None, None

        self.out_type = self.conv.out_type

    def forward(self, x):
        return self.conv(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantNorm(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        num_groups: int = None,
        affine: bool = False,
    ):
        super().__init__()
        self.in_type = in_type

        if num_groups:  # group norm
            norm = GroupNorm
            induced_norm = GroupStandardization
            param = num_groups
        else:  # batch norm
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


class EquivariantConvBlock(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        frequency: int = None,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        num_groups: int = None,
        pool_size: int = None,
        invariant_map: bool = False,
    ):
        super().__init__()
        self.in_type = in_type  # declaration required by base class
        self.conv = EquivariantConv(
            self.in_type,
            out_channels,
            frequency=frequency,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        # Induced
        if self.in_type.gspace.fibergroup.name == "O(2)":
            labels = ["trivial"] * len(self.conv.trivials) + ["gate"] * len(
                self.conv.gate
            )
            modules = [
                (Mish(self.conv.trivials), "trivial"),
                (InducedGatedNonLinearity(self.conv.gate), "gate"),
            ]
            self.act_func = MultipleModule(self.conv.out_type, labels, modules)
        # Cyclic and Dihedral Groups
        else:
            self.act_func = Mish(self.conv.out_type)

        self.norm = EquivariantNorm(
            self.act_func.out_type, num_groups=num_groups, affine=False
        )

        if pool_size is not None or invariant_map:
            self.pool = EquivariantPool(self.norm.out_type, pool_size, invariant_map)
            self.out_type = self.pool.out_type
        else:
            self.pool = nn.Identity()
            self.out_type = self.norm.out_type

    def forward(self, x):
        x = self.conv(x)
        x = self.act_func(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantWideConvBlock(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        frequency: int = None,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        num_groups: int = None,
    ):
        super(EquivariantWideConvBlock, self).__init__()
        self.in_type = in_type

        self.conv1 = EquivariantConv(
            self.in_type,
            out_channels,
            frequency=frequency,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            dilation=dilation,
            bias=bias,
        )

        # Induced
        if self.in_type.gspace.fibergroup.name == "O(2)":
            labels = ["trivial"] * len(self.conv1.trivials) + ["gate"] * len(
                self.conv1.gate
            )
            modules = [
                (Mish(self.conv1.trivials), "trivial"),
                (InducedGatedNonLinearity(self.conv1.gate), "gate"),
            ]
            self.act_func1 = MultipleModule(self.conv1.out_type, labels, modules)
        # Cyclic and Dihedral Groups
        else:
            self.act_func1 = Mish(self.conv1.out_type)

        self.norm1 = EquivariantNorm(
            self.act_func1.out_type, num_groups=num_groups, affine=False
        )

        self.conv2 = EquivariantConv(
            self.norm1.out_type,
            out_channels,
            frequency=frequency,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        # Induced
        if self.in_type.gspace.fibergroup.name == "O(2)":
            labels = ["trivial"] * len(self.conv1.trivials) + ["gate"] * len(
                self.conv1.gate
            )
            modules = [
                (Mish(self.conv1.trivials), "trivial"),
                (InducedGatedNonLinearity(self.conv1.gate), "gate"),
            ]
            self.act_func2 = MultipleModule(self.conv2.out_type, labels, modules)
        # Cyclic and Dihedral Groups
        else:
            self.act_func2 = Mish(self.conv2.out_type)

        self.norm2 = EquivariantNorm(
            self.act_func2.out_type, num_groups=num_groups, affine=False
        )

        self.out_type = self.norm2.out_type

        self.shortcut = nn.Identity()
        if stride != 1 or self.in_type != self.out_type:
            shortcut = EquivariantConv(
                self.in_type,
                out_channels,
                frequency=frequency,
                kernel_size=1,
                padding=0,
                stride=stride,
                dilation=dilation,
                bias=bias,
                no_gates=True,
            )
            norm = EquivariantNorm(
                shortcut.out_type, num_groups=num_groups, affine=False
            )
            self.shortcut = SequentialModule(*[shortcut, norm])

    def forward(self, x):
        out = self.conv1(x)
        out = self.act_func1(out)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.act_func2(out)
        out = self.norm2(out)
        out += self.shortcut(x)
        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantSqueezeExcitation(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        in_channels: int,
        squeeze_channels: int,
        frequency: int = None,
    ):
        super(EquivariantSqueezeExcitation, self).__init__()
        self.in_type = in_type

        self.avgpool = PointwiseAdaptiveAvgPool(self.in_type, 1)
        self.fc1 = EquivariantConv(
            self.avgpool.out_type, squeeze_channels, frequency, kernel_size=1, padding=0
        )

        # Induced
        if self.in_type.gspace.fibergroup.name == "O(2)":
            labels = ["trivial"] * len(self.fc1.trivials) + ["gate"] * len(
                self.fc1.gate
            )
            modules = [
                (Mish(self.fc1.trivials), "trivial"),
                (InducedGatedNonLinearity(self.fc1.gate), "gate"),
            ]
            self.activation = MultipleModule(self.fc1.out_type, labels, modules)
        # Cyclic and Dihedral Groups
        else:
            self.activation = Mish(self.fc1.out_type)

        self.fc2 = EquivariantConv(
            self.activation.out_type, in_channels, frequency, kernel_size=1, padding=0
        )

        # TODO: Check if trivial, regular and induced reps support norm non-linearity
        self.scale_activation = NormNonLinearity(
            self.fc2.out_type, function="n_sigmoid"
        )

        self.out_type = self.scale_activation.out_type

    def _scale(self, input: GroupTensor):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: GroupTensor):
        scale = self._scale(input)
        return GroupTensor(scale.tensor * input.tensor, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
