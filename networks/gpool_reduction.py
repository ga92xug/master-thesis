from nn import GSpace2D, FieldType, GroupTensor

from nn.modules.equivariant_module import EquivariantModule
from nn.modules.utils import indexes_from_labels

import torch
from torch import nn

from typing import List, Tuple, Any
from collections import defaultdict
import numpy as np
import re

class GroupPoolingReduction(EquivariantModule):
    def __init__(self, in_type: FieldType, **kwargs):
        r"""

        Module that implements *group pooling*.
        This module only supports permutation representations such as regular representation,
        quotient representation or trivial representation (though, in the last case, this module
        acts as identity).
        For each input field, an output field is built by taking the maximum activation within that field; as a result,
        the output field transforms according to a trivial representation.

        .. seealso::
            :attr:`~Group.regular_representation`,
            :attr:`~Group.quotient_representation`

        Args:
            in_type (FieldType): the input field type

        """
        assert isinstance(in_type.gspace, GSpace2D)

        for r in in_type.representations:
            assert (
                "pointwise" in r.supported_nonlinearities
            ), 'Error! Representation "{}" does not support "pointwise" non-linearity'.format(
                r.name
            )

        super(GroupPoolingReduction, self).__init__()

        self.reduction_factor = calculate_reduction_factor(old_group=parse_groups(str(in_type)), new_group=str(in_type.fibergroup))
        assert in_type.size % self.reduction_factor == 0, "Error! size must divide the number of channels"
        index_size = in_type.size // self.reduction_factor

        self.space = in_type.gspace
        self.in_type = in_type

        # build the output representation substituting each input field with a trivial representation
        #self.out_type = FieldType(self.space, [self.space.trivial_repr] * len(in_type))
        self.out_type = FieldType(self.space, [self.space.regular_repr] * len(in_type))
        # print("self.in_type: ", self.in_type, "len(self.in_type): ", len(self.in_type), "self.in_type.size: ", self.in_type.size)

        # indices of the channels corresponding to fields belonging to each group in the input representation
        _in_indices = defaultdict(list)
        # indices of the channels corresponding to fields belonging to each group in the output representation
        _out_indices = defaultdict(list)

        # group fields by their size and retrieve the indices of the fields
        indeces = indexes_from_labels(
            in_type, [r.size for r in in_type.representations]
        )

        self.in_indices = {}
        self.out_indices = {}


        for s, (fields, idxs) in indeces.items():
            _in_indices[s] = torch.LongTensor([min(idxs), max(idxs) + 1])
            # _out_indices[s] = torch.LongTensor([min(fields), max(fields) + 1])
            _out_indices[s] = torch.LongTensor([0, index_size])

            # register the indices tensors as parameters of this module
            self.in_indices[s] = _in_indices[s].to(
                f"cuda:{torch.cuda.current_device()}"
            )
            self.out_indices[s] = _out_indices[s].to(
                f"cuda:{torch.cuda.current_device()}"
            )


    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Apply Group Pooling to the input feature map.

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        #if self.reduction_factor == 1:
        #    return input

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]
        #print("coords: ", coords, "b: ", b, "c: ", c, "spatial_shape: ", spatial_shape, "input.shape: ", input.shape)
        #print("input.shape: ", input.shape)

        output = torch.empty(
            self.evaluate_output_shape(input.shape),
            device=input.device,
            dtype=torch.float,
        )

        for s in list(self.in_indices.keys()):
            in_indices = self.in_indices[s]
            fm = input[:, in_indices[0] : in_indices[1], ...]
            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)
            _, channels, _, _, _ = fm.shape
            fm = fm.view(b, channels, s//self.reduction_factor, -1, *spatial_shape)
            #print("fm.shape: ", fm.shape)
            #print("torch.max(fm, 3)[0].shape: ", torch.max(fm, 3)[0].view(b, -1, *spatial_shape).shape)

            output = torch.max(fm, 3)[0].view(b, -1, *spatial_shape)
            #output = output.view(b, -1, *spatial_shape)

        # wrap the result in a GroupTensor
        return GroupTensor(output, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        c = self.in_type.size

        x = torch.randn(3, c, 10, 10)

        x = GroupTensor(x, self.in_type)

        errors = []

        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))

            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(
                f"Group {el}: err max: {errs.max()} - err mean: {errs.mean()} - err var: {errs.var()}"
            )

            assert torch.allclose(
                out1.tensor, out2.tensor, atol=atol, rtol=rtol
            ), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                el, errs.max(), errs.mean(), errs.var()
            )

            errors.append((el, errs.mean()))

        return errors

    def export(self):
        r"""
        Export this module to the pure PyTorch module :class:`~nn.MaxPoolChannels`
        and set to "eval" mode.

        .. warning ::

                Currently, this method only supports group pooling with feature types containing only representations
                of the same size.

        .. note ::

            Because there is no native PyTorch module performing this operation, it is not possible to export this
            module without any dependency with this library.
            Indeed, the resulting module is dependent on this library through the class
            :class:`~nn.MaxPoolChannels`.
            In case PyTorch will introduce a similar module in a future release, we will update this method to remove
            this dependency.

            Nevertheless, the :class:`~nn.MaxPoolChannels` module is slightly lighter
            than :class:`~nn.GroupPooling` as it does not perform any automatic type checking and does not wrap
            each tensor in a :class:`~nn.GroupTensor`.
            Furthermore, the :class:`~nn.MaxPoolChannels` class is very simple and
            one can easily reimplement it to remove any dependency with this library after training the model.

        """

        if len(self.in_indices) > 1:
            raise NotImplementedError(
                """
                Group pooling with feature types containing representations of different sizes is not supported yet.
            """
            )

        self.eval()

        size = int(list(self.in_indices.keys())[0])
        gpool = MaxPoolChannels(size)

        return gpool.eval()

    def extra_repr(self):
        return "{in_type}".format(**self.__dict__)


def calculate_reduction_factor(old_group: str, new_group: str):
    #print("old_group: ", old_group, "new_group: ", new_group)
    old_letter = old_group[0]  # First character represents the letter
    old_number = int(old_group[1:])  # Remaining characters represent the number

    new_letter = new_group[0]
    new_number = int(new_group[1:])

    assert old_number % new_number == 0, "The new group must be a subgroup of the old group"
    assert new_letter in ["C", "D"] and old_letter in ["C", "D"], "The group must be either C or D"
    if old_letter == "C":
        assert new_letter != "D", "The new group must be a subgroup of the old group"
    numerical_reduction = old_number // new_number

    if old_letter == 'D' and new_letter == 'C':
        letter_reduction = 2
    else:
        letter_reduction = 1

    overall_reduction_factor = numerical_reduction * letter_reduction
    return overall_reduction_factor

def parse_groups(string):
    new_group_pattern = r"\[(\w\d+)"
    old_group_pattern = r": \{(\w\d+)"

    new_group_match = re.search(new_group_pattern, string)
    old_group_match = re.search(old_group_pattern, string)

    if new_group_match:
        new_group = new_group_match.group(1)
    else:
        raise ValueError("New group not found in the string.")

    old_group = old_group_match.group(1) if old_group_match else None

    return old_group