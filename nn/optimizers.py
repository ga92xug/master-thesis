# TODO: Switch to Objax

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Union

import torch
from torch.optim import Optimizer

from opacus.optimizers import DPOptimizer


logger = logging.getLogger(__name__)


class AdaClipDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    adaptive clipping strategy
    https://arxiv.org/pdf/1905.03871.pdf
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        target_unclipped_quantile: float,
        clipbound_learning_rate: float,
        max_clipbound: float,
        min_clipbound: float,
        unclipped_num_std: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        assert (
            max_clipbound > min_clipbound
        ), "max_clipbound must be larger than min_clipbound."
        self.target_unclipped_quantile = target_unclipped_quantile
        self.clipbound_learning_rate = clipbound_learning_rate
        self.max_clipbound = max_clipbound
        self.min_clipbound = min_clipbound
        self.unclipped_num_std = unclipped_num_std
        # Theorem 1. in  https://arxiv.org/pdf/1905.03871.pdf
        self.noise_multiplier = (
            self.noise_multiplier ** (-2) - (2 * unclipped_num_std) ** (-2)
        ) ** (-1 / 2)
        self.sample_size = 0
        self.unclipped_num = 0

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients, self.sample_size and self.unclipped_num
        """
        super().zero_grad(set_to_none)

        self.sample_size = 0
        self.unclipped_num = 0

    def clip_and_accumulate(self):
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        # the two lines below are the only changes
        # relative to the parent DPOptimizer class.
        self.sample_size += len(per_sample_clip_factor)
        self.unclipped_num += (
            len(per_sample_clip_factor) - (per_sample_clip_factor < 1).sum()
        )

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        super().add_noise()

        unclipped_num_noise = _generate_noise(
            std=self.unclipped_num_std,
            reference=self.unclipped_num,
            generator=self.generator,
        )

        self.unclipped_num = float(self.unclipped_num)
        self.unclipped_num += unclipped_num_noise

    def update_max_grad_norm(self):
        """
        Update clipping bound based on unclipped fraction
        """
        unclipped_frac = self.unclipped_num / self.sample_size
        self.max_grad_norm *= torch.exp(
            -self.clipbound_learning_rate
            * (unclipped_frac - self.target_unclipped_quantile)
        )
        if self.max_grad_norm > self.max_clipbound:
            self.max_grad_norm = self.max_clipbound
        elif self.max_grad_norm < self.min_clipbound:
            self.max_grad_norm = self.min_clipbound

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        pre_step_full = super().pre_step()
        if pre_step_full:
            self.update_max_grad_norm()
        return pre_step_full


# Functions from opacus.optimizers.optimier
def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.
    DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
    no gradient can be used twice - client must call .zero_grad() between
    optimizer steps, otherwise privacy guarantees are compromised.
    This method marks tensors that have already been used in optimizer steps to then
    check if zero_grad has been duly called.
    Notes:
          This is used to only mark ``p.grad_sample`` and ``p.summed_grad``
    Args:
        obj: tensor or a list of tensors to be marked
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.
    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`
    Args:
        x: gradient tensor
    Raises:
        ValueError
            If tensor has attribute ``._processed`` previously set by
            ``_mark_as_processed`` method
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously
    used in optimization step.
    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`
    Args:
        x: gradient tensor or a list of tensors
    Raises:
        ValueError
            If tensor (or at least one tensor from the list) has attribute
            ``._processed`` previously set by ``_mark_as_processed`` method
    """

    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)


def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0
    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generate
            (see the notes)
    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).
        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )


def _get_flat_grad_sample(p: torch.Tensor):
    """
    Return parameter's per sample gradients as a single tensor.
    By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
    batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
    only one batch, and a list of tensors if gradients are accumulated over multiple
    steps. This is done to provide visibility into which sample belongs to which batch,
    and how many batches have been processed.
    This method returns per sample gradients as a single concatenated tensor, regardless
    of how many batches have been accumulated
    Args:
        p: Parameter tensor. Must have ``grad_sample`` attribute
    Returns:
        ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
        concatenating every tensor in ``p.grad_sample`` if it's a list
    Raises:
        ValueError
            If ``p`` is missing ``grad_sample`` attribute
    """

    if not hasattr(p, "grad_sample"):
        raise ValueError(
            "Per sample gradient not found. Are you using GradSampleModule?"
        )
    if p.grad_sample is None:
        raise ValueError(
            "Per sample gradient is not initialized. Not updated in backward pass?"
        )
    if isinstance(p.grad_sample, torch.Tensor):
        return p.grad_sample
    elif isinstance(p.grad_sample, list):
        return torch.cat(p.grad_sample, dim=0)
    else:
        raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")
