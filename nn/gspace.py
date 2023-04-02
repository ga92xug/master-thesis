from __future__ import annotations

import nn
from group_theory import (
    Group,
    GroupElement,
    Representation,
    IrreducibleRepresentation,
    KernelBasis,
    O2,
    Representation,
    KernelBasis,
    kernel2d_so2,
    kernel2d_o2,
    kernel2d_o2_subgroup,
    kernel2d_so2_subgroup,
)

from .utils import linear_transform_array_nd

from abc import ABC, abstractmethod
from typing import Tuple, Callable, List, Union

from collections import defaultdict

import numpy as np


__all__ = [
    "GSpace2D",
    #################
    "rot2dOnR2",
    "flipRot2dOnR2",
    "flip2dOnR2",
    "trivialOnR2",
]


class GSpace2D(ABC):
    def __init__(self, sg_id: Tuple, maximum_frequency: int = 6):
    #def __init__(self, fibergroup: Group, dimensionality: int, name: str):
        r"""
        Stefan: GSpace description
        
        Abstract class for G-spaces.

        A ``GSpace`` describes the space where a signal lives (e.g. :math:`\R^2` for planar images) and its symmetries
        (e.g. rotations or reflections).
        As an `Euclidean` base space is assumed, a G-space is fully specified by the ``dimensionality`` of the space
        and a choice of origin-preserving symmetry group (``fibergroup``).

        .. seealso::

            :class:`~GSpace0D`,
            :class:`~GSpace2D`,
            or the factory methods
            :class:`~flipRot2dOnR2`,
            :class:`~rot2dOnR2`,
            :class:`~flip2dOnR2`,
            :class:`~trivialOnR2`

        .. note ::

            Mathematically, this class describes a *Principal Bundle*
            :math:`\pi : (\R^D, +) \rtimes G \to \mathbb{R}^D, tg \mapsto tG`,
            with the Euclidean space :math:`\mathbb{R}^D` (where :math:`D` is the ``dimensionality``) as `base space`
            and :math:`G` as `fiber group` (``fibergroup``).
            For more details on this interpretation we refer to
            `A General Theory of Equivariant CNNs On Homogeneous Spaces <https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf>`_.


        Args:
            fibergroup (Group): the fiber group
            dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            name (str): an identification name

        Attributes:
            ~.fibergroup (Group): the fiber group
            ~.dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            ~.name (str): an identification name
            ~.basespace (str): the name of the space whose symmetries are modeled. It is an Euclidean space :math:`\R^D`.

        -----------------
        Stefan: GSpace2D description

        Describes reflectional and rotational symmetries of the plane :math:`\R^2`.

        .. note ::
            A point :math:`\bold{v} \in \R^2` is parametrized using an :math:`(X, Y)` convention,
            i.e. :math:`\bold{v} = (x, y)^T`.
            The representation :attr:`GSpace2D.basespace_action` also assumes this convention.

            However, when working with data on a pixel grid, the usual :math:`(-Y, X)` convention is used.
            That means that, in a 4-dimensional feature tensor of shape ``(B, C, D1, D2)``, the last dimension
            is the X axis while the second last is the (inverted) Y axis.
            Note that this is consistent with 2D images, where a :math:`(-Y, X)` convention is used.



        """

        o2 = O2(maximum_frequency)
        _sg_id = o2._process_subgroup_id(sg_id)
        fibergroup, inclusion, restriction = o2.subgroup(_sg_id)

        # TODO - catch sg_id and build a dictionary of more meaningful names
        # use the input sg_id instead of the processed one to avoid adding the adjoint parameter unless specified
        name = f"{fibergroup}_on_R2[{sg_id}]"

        self._sg_id = _sg_id
        self._inclusion = inclusion
        self._restriction = restriction
        self._base_action = o2.irrep(1, 1).restrict(_sg_id)

        dimensionality = 2
        # Stefan: from here on original GSpace code

        # TODO move this sub-package to PyTorch

        self.name = name
        self.dimensionality = dimensionality
        self.fibergroup = fibergroup
        self.basespace = f"R^{self.dimensionality}"

        # To not recompute the basis for the same intertwiner as many times as it appears,
        # the basis is stored in these dictionaries the first time we compute it

        # Store the computed intertwiners between irreps
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_irrep, output_irrep) pairs to the corresponding basis
        self._irreps_intertwiners_basis_memory = defaultdict(lambda: dict())

        # Store the computed intertwiners between general representations
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_repr, output_repr) pairs to the corresponding basis
        self._fields_intertwiners_basis_memory = defaultdict(dict)

        # Store the computed intertwiners between general representations
        # - key = (input_repr, output_repr)
        # - value = the corresponding basis
        self._fields_intertwiners_basis_memory_fiber_basis = dict()

    def type(self, *representations: Representation) -> nn.FieldType:
        f"""
        Shortcut to build a :class:`~nn.FieldType`.
        This is equivalent to ```FieldType(gspace, representations)```.
        """
        return nn.FieldType(self, representations)

    
    def restrict(self, id: Tuple) -> Tuple[GSpace2D, Callable, Callable]:
        r"""

        Build the :class:`~GSpace` associated with the subgroup of the current fiber group identified by
        the input ``id``.
        This reduces the level of symmetries of the base space to be considered.

        Check the ``restrict`` method's documentation in the non-abstract subclass used for a description of the
        parameter ``id``.

        Args:
            id: id of the subgroup

        Returns:
            a tuple containing

                - **gspace**: the restricted gspace

                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space

                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the subgroup (returns ``None`` if the element is not in the subgroup)

        ----
        Stefan: GSpace2D description

        Build the :class:`~group.GSpace` associated with the subgroup of the current fiber group identified by
        the input ``id``.

        Args:
            id (tuple): the id of the subgroup

        Returns:
            A tuple containing:
                - **gspace**: the restricted gspace
                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space
                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the
                  subgroup (returns ``None`` if the element is not in the subgroup)
        
        """

        o2 = O2()
        sg_id = o2._combine_subgroups(self._sg_id, id)
        sg, inclusion, restriction = self.fibergroup.subgroup(id)

        return GSpace2D(sg_id), inclusion, restriction

    @property
    def basespace_action(self) -> Representation:
        r"""

        Defines how the fiber group transforms the base space.

        More precisely, this method defines how an element :math:`g \in G` of the fiber group transforms a point
        :math:`x \in X \cong \R^d` of the base space.
        This action is defined as a :math:`d`-dimensional linear :class:`~Representation` of :math:`G`.

        """
        return self._base_action

    def _interpolate_transform_basespace(
        self,
        input: np.ndarray,
        element: GroupElement,
        order: int = 2,
    ) -> np.ndarray:
        r"""

        Defines how the fiber group transforms the base space.

        The methods takes a tensor compatible with this space (i.e. whose spatial dimensions are supported by the
        base space) and returns the transformed tensor.

        More precisely, given an input tensor, interpreted as an :math:`n`-dimensional signal
        :math:`f: X \to \mathbb{R}^n` defined over the base space :math:`X`, and an element :math:`g \in G` of the
        fiber group, the methods return the transformed signal :math:`f'` defined as:

        .. math::
            f'(x) := f(g^{-1} x)

        This method is specific of the particular GSpace and defines how :math:`g^{-1}` transforms a point
        :math:`x \in X` of the base space.


        Args:
            input (~numpy.ndarray): input tensor
            element (GroupElement): element of the fiber group

        Returns:
            the transformed tensor

        """
        assert element.group == self.fibergroup
        action = self.basespace_action
        trafo = action(element)
        return linear_transform_array_nd(input, trafo, order=order)

    @property
    def irreps(self) -> List[IrreducibleRepresentation]:
        r"""
        list containing all the already built irreducible representations of the fiber group of this space.

        .. seealso::

            See :attr:`Group.irreps` for more details

        """
        return self.fibergroup.irreps()

    @property
    def representations(self):
        r"""
        Dictionary containing all the already built representations of the fiber group of this space.

        .. seealso::

            See :attr:`Group.representations` for more details

        """
        return self.fibergroup.representations

    @property
    def trivial_repr(self) -> Representation:
        r"""
        The trivial representation of the fiber group of this space.

        .. seealso::

            :attr:`Group.trivial_representation`

        """
        return self.fibergroup.trivial_representation

    def irrep(self, *id) -> IrreducibleRepresentation:
        r"""
        Builds the irreducible representation (:class:`~IrreducibleRepresentation`) of the fiber group
        identified by the input arguments.

        .. seealso::

            This method is a wrapper for :meth:`Group.irrep`. See its documentation for more details.
            Check the documentation of :meth:`~Group.irrep` of the specific fiber group used for more
            information on the valid ``id``.


        Args:
            *id: parameters identifying the irrep.

        """
        return self.fibergroup.irrep(*id)

    @property
    def regular_repr(self) -> Representation:
        r"""
        The regular representation of the fiber group of this space.

        .. seealso::

            :attr:`Group.regular_representation`

        """
        return self.fibergroup.regular_representation

    def induced_repr(self, subgroup_id, repr: Representation) -> Representation:
        r"""
        Builds the induced representation of the fiber group of this space from the representation ``repr`` of
        the subgroup identified by ``subgroup_id``.

        Check the :meth:`~GSpace.restrict` method's documentation in the non-abstract subclass used
        for a description of the parameter ``subgroup_id``.

        .. seealso::

            See :attr:`Group.induced_representation` for more details on the representation.

        Args:
            subgroup_id: identifier of the subgroup
            repr (Representation): the representation of the subgroup to induce

        """
        return self.fibergroup.induced_representation(subgroup_id, repr)

    @property
    def testing_elements(self):
        return self.fibergroup.testing_elements()

    def build_kernel_basis(
        self,
        in_repr: Representation,
        out_repr: Representation,
        sigma: Union[float, List[float]],
        rings: List[float],
        **kwargs,
    ) -> KernelBasis:
        r"""
        Builds a basis for the space of the equivariant kernels with respect to the symmetries described by this
        :class:`~GSpace`.

        A kernel :math:`\kappa` equivariant to a group :math:`G` needs to satisfy the following equivariance constraint:

        .. math::
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1}  \qquad \forall g \in G, x \in \R^D

        where :math:`\rho_\text{in}` is ``in_repr`` while :math:`\rho_\text{out}` is ``out_repr``.


        Because the equivariance constraints only restrict the angular part of the kernels, any radial profile is
        permitted.
        The basis for the radial profile used here contains rings with different radii (``rings``)
        associated with (possibly different) widths (``sigma``).
        A ring is implemented as a Gaussian function over the radial component, centered at one radius
        (see also :class:`~GaussianRadialProfile`).

        .. note ::
            This method is a wrapper for the functions building the bases which are defined in :doc:`kernels`:
            - :meth:`kernel2d_o2`,
            - :meth:`kernel2d_so2`

        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            sigma (list or float): parameters controlling the width of each ring of the radial profile.
                    If only one scalar is passed, it is used for all rings
            rings (list): radii of the rings defining the radial profile
            **kwargs: Group-specific keywords arguments for ``_basis_generator`` method

        Returns:
            an instance of :class:`~KernelBasis` representing the analytical basis

        """
        assert isinstance(in_repr, Representation)
        assert isinstance(out_repr, Representation)

        assert in_repr.group == self.fibergroup
        assert out_repr.group == self.fibergroup

        if isinstance(sigma, float):
            sigma = [sigma] * len(rings)

        assert all([s > 0.0 for s in sigma])
        assert len(sigma) == len(rings)

        # build the key
        key = dict(**kwargs)
        key["sigma"] = tuple(sigma)
        key["rings"] = tuple(rings)
        key = tuple(sorted(key.items()))

        if (in_repr.name, out_repr.name) not in self._fields_intertwiners_basis_memory[
            key
        ]:
            # TODO - we could use a flag in the args to choose whether to store it or not

            basis = self._basis_generator(in_repr, out_repr, rings, sigma, **kwargs)

            # store the basis in the dictionary
            self._fields_intertwiners_basis_memory[key][
                (in_repr.name, out_repr.name)
            ] = basis

        # return the dictionary with all the basis built for this filter size
        return self._fields_intertwiners_basis_memory[key][
            (in_repr.name, out_repr.name)
        ]

    
    def _basis_generator(
        self,
        in_repr: Representation,
        out_repr: Representation,
        rings: List[float],
        sigma: List[float],
        **kwargs,
    ) -> KernelBasis:
        r"""
        Method that builds the analytical basis that spans the space of equivariant filters which
        are intertwiners between the representations induced from the representation ``in_repr`` and ``out_repr``.

        `kwargs` can be used to specify `maximum_frequency`

        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            rings (list): radii of the rings where to sample the bases
            sigma (list): parameters controlling the width of each ring where the bases are sampled.

        Returns:
            the basis built

        """

        # TODO - add max_offset for cyclic and dihedral groups!

        if "maximum_frequency" in kwargs:
            maximum_frequency = kwargs["maximum_frequency"]
        else:
            maximum_frequency = None

        if self._sg_id[0] is not None and self._sg_id[1] == -1:
            return kernel2d_o2(
                in_repr,
                out_repr,
                rings,
                sigma,
                axis=self._sg_id[0] / 2,
                maximum_frequency=maximum_frequency,
                filter=None,
            )
        elif self._sg_id == (None, -1):
            return kernel2d_so2(
                in_repr,
                out_repr,
                rings,
                sigma,
                maximum_frequency=maximum_frequency,
                filter=None,
            )
        elif self._sg_id[0] is None:
            sg_id = self._sg_id[1]
            return kernel2d_so2_subgroup(
                in_repr,
                out_repr,
                sg_id,
                rings,
                sigma,
                adjoint=None,
                maximum_frequency=maximum_frequency,
                filter=None,
            )
        else:
            return kernel2d_o2_subgroup(
                in_repr,
                out_repr,
                self._sg_id,
                rings,
                sigma,
                axis=0.0,
                adjoint=None,
                maximum_frequency=maximum_frequency,
                filter=None,
            )

    def __repr__(self):
        return self.name
    
    @property
    def rotations_order(self):
        return self._sg_id[1]
    
    @property
    def flips_order(self):
        return 1 if self._sg_id[0] is not None else 0
    
    def __eq__(self, other):
        if isinstance(other, GSpace2D):
            return self._sg_id == other._sg_id
        else:
            return False

    def __hash__(self):
        return 1000 * hash(self.name) + hash(self._sg_id)



def rot2dOnR2(N: int = -1, maximum_frequency: int = 6) -> GSpace2D:
    r"""

    Describes rotation symmetries of the plane :math:`\R^2`.

    If ``N > 1``, the gspace models *discrete* rotations by angles which are multiple of :math:`\frac{2\pi}{N}`
    (:class:`~e2cnn.group.CyclicGroup`).
    Otherwise, if ``N=-1``, the gspace models *continuous* planar rotations (:class:`~e2cnn.group.SO2`).
    In that case the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
    :class:`~e2cnn.group.SO2` (see its documentation for more details)

    Args:
        N (int): number of discrete rotations (integer greater than 1) or ``-1`` for continuous rotations
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.SO2`'s irreps if ``N = -1``

    """
    assert isinstance(N, int)
    assert N == -1 or N > 0
    sg_id = None, N
    return GSpace2D(sg_id, maximum_frequency=maximum_frequency)


def flipRot2dOnR2(
    N: int = -1, maximum_frequency: int = 6, axis: float = np.pi / 2.0
) -> GSpace2D:
    r"""
    Describes reflectional and rotational symmetries of the plane :math:`\R^2`.

    Reflections are applied with respect to the line through the origin with an angle ``axis`` degrees with respect
    to the *X*-axis.

    If ``N > 1``, this gspace models reflections and *discrete* rotations by angles multiple of :math:`\frac{2\pi}{N}`
    (:class:`~e2cnn.group.DihedralGroup`).
    Otherwise, if ``N=-1`` (by default), the class models reflections and *continuous* planar rotations
    (:class:`~e2cnn.group.O2`).
    In that case, the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
    :class:`~e2cnn.group.O2` (see its documentation for more details)

    .. note ::

        All axes obtained from the axis defined by ``axis`` with a rotation in the symmetry group are equivalent.
        For instance, if ``N = 4``, an axis :math:`\beta` is equivalent to the axis :math:`\beta + \pi/2`.
        It follows that for ``N = -1``, i.e. in case the symmetry group contains all continuous rotations, any
        reflection axis is theoretically equivalent.
        In practice, though, a basis for equivariant convolutional filter sampled on a grid is affected by the
        specific choice of the axis. In general, choosing an axis aligned with the grid (an horizontal or a
        vertical axis, i.e. :math:`0` or :math:`\pi/2`) is suggested.

    Args:
        N (int): number of discrete rotations (integer greater than 1) or -1 for continuous rotations
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.O2` 's irreps if ``N = -1``
        axis (float, optional): the slope of the axis of the flip (in radians)

    """

    assert isinstance(N, int)
    assert N == -1 or N > 0
    sg_id = 2 * axis, N
    return GSpace2D(sg_id, maximum_frequency=maximum_frequency)


def flip2dOnR2(axis: float = np.pi / 2) -> GSpace2D:
    r"""

    Describes reflectional symmetries of the plane :math:`\R^2`.

    Reflections are applied along the line through the origin with an angle ``axis`` degrees with respect to
    the *X*-axis.

    Args:
        axis (float, optional): the slope of the axis of the reflection (in radians).
                                By default, the vertical axis is used (:math:`\pi/2`).

    """
    sg_id = 2 * axis, 1
    return GSpace2D(sg_id, maximum_frequency=1)


def trivialOnR2() -> GSpace2D:
    r"""
    Describes the plane :math:`\R^2` without considering any origin-preserving symmetry.
    This is modeled by choosing trivial fiber group :math:`\{e\}`.

    .. note ::
        This models the symmetries of conventional *Convolutional Neural Networks* which are not equivariant to
        origin preserving transformations such as rotations and reflections.

    """
    sg_id = (None, 1)
    return GSpace2D(sg_id, maximum_frequency=1)
