import torch
import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Self, Tuple, Union

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


class Own_RandomRotation(torch.nn.Module):
    """
    normal random rotation does not allow for discrete choices.

    Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.

    """

    def __init__(self, degrees, interpolation=InterpolationMode.BILINEAR, expand=False):
        super().__init__()
        
        self.degrees = list(degrees)
        self.interpolation = interpolation
        self.expand = expand


    def get_params(self) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """

        # make random choice of values in degrees
        angle = random.choice(self.degrees)
        return angle


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = 0
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params()

        return F.rotate(img, angle, self.interpolation, self.expand, fill=fill)


    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(degrees={self.degrees}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", expand={self.expand}"
        format_string += ")"
        return format_string