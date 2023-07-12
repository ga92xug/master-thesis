from copy import deepcopy
import math
from typing import Tuple
import warnings
import sys
import torch
sys.path.append('../scaling-laws-ecnn') # add parent directory

from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
)

CHANNELS_CONSTANT = 1

def cuda_memory_usage(verbose=1):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    if verbose >= 1:
        print(f"Used: {r / 1024 ** 3:.1f}/{t / 1024 ** 3:.1f} GB")
        print(f"Allocated:    {a / 1024 ** 3:.1f} GB")
        # print(f"Free:         {f / 1024 ** 3:.1f} GB")
    return r / t

def get_group_id(reflection, group):
    return (reflection, group) if reflection >= 0 else (None, group)

def get_width_and_height_from_size(x):
    """Obtain height and width from x.
    Args:
        x (int, tuple or list): Data size.
    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    if isinstance(x, str):
        return int(x), int(x)
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.
    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.
    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def get_fixed_params_old(type_equi_block, fix_params_mode, normal_block=None, gspace=None,
                   channel_name="out_channels", **kwargs):
    """
    Legacy function for finding an equivariant convolutional block with a parameter count similar to a normal
    """
    N = gspace.fibergroup.order()
    kwargs[channel_name] = int(kwargs[channel_name] / N)
    if fix_params_mode in ["heuristic", "all"]:
        kwargs[channel_name] = int(kwargs[channel_name] * math.sqrt(N * CHANNELS_CONSTANT))

    equi_block = type_equi_block(**kwargs)
    if fix_params_mode in ["heuristic", "no"]:
        return equi_block
    
    # fix param iter search
    param_normal_block = get_param_count(normal_block)
    param_equi_block = get_param_count(equi_block)
    out_channels = kwargs[channel_name]
    old_equi_param = None
    # initialize search range
    if param_equi_block > param_normal_block:
        lower_bound = max(out_channels - 400, 1)
        upper_bound = out_channels
    else:
        lower_bound = out_channels
        # the upper bound search is expensive so we gradually increase it
        upper_bound = int(round(out_channels // 0.5) + 100)
    # binary search
    while lower_bound <= upper_bound:
        # print(f'lower bound: {lower_bound}, upper bound: {upper_bound}, prediction: {kwargs[channel_name]}')
        kwargs[channel_name] = (lower_bound + upper_bound) // 2
        # save the old one since we might not be in 1% range
        old_equi_param, old_equi_conv_block = param_equi_block, equi_block 
        # get new equi_block
        equi_block = type_equi_block(**kwargs)
        param_equi_block = get_param_count(equi_block)
        if abs(param_equi_block - param_normal_block) < 0.01:
            last_ratio = param_equi_block / param_normal_block
            print(f'Ratio for block: {last_ratio:.3f}')
            return equi_block
        if param_equi_block < param_normal_block:
            # prediction is too small
            lower_bound = kwargs[channel_name] + 1
            # if lower_bound >= upper_bound:
            #      # we increase to upper bound slowly to avoid expensive search
            #      upper_bound *= 2
                 
        else:
            upper_bound = kwargs[channel_name] - 1
                
    # if no solution found, return closest channel size
    if old_equi_param is not None:
        if abs(old_equi_param - param_normal_block) < abs(param_equi_block - param_normal_block):
            equi_block = old_equi_conv_block
        
    last_ratio = param_equi_block / param_normal_block
    print(f'Ratio for block: {last_ratio:.3f}')
    return equi_block

def get_fixed_params(type_equi_block, fix_params_mode, normal_block=None, gspace=None,
                     channel_name="out_channels", max_iterations=50, **kwargs):
    """
    Find an equivariant convolutional block with a parameter count similar to a normal
    convolutional block using binary search or a heuristic approach.

    Args:
        type_equi_block (type): The type of the equivariant convolutional block.
        fix_params_mode (str): The mode to use for fixing parameters ('heuristic', 'all', or 'no').
        normal_block (type, optional): The type of the normal convolutional block.
        gspace (object, optional): The symmetry group for the equivariant block.
        channel_name (str, optional): The name of the channel parameter. Default is 'out_channels'.
        **kwargs: Additional keyword arguments for the equivariant convolutional block.

    Returns:
        equi_block: An equivariant convolutional block with a parameter count similar to
                    the normal convolutional block.
    """
    out_channels = kwargs[channel_name]
    N = gspace.fibergroup.order()
    kwargs[channel_name] = kwargs[channel_name] / N
    if fix_params_mode in ["heuristic", "all"]:
        kwargs[channel_name] = kwargs[channel_name] * math.sqrt(N * CHANNELS_CONSTANT)

    kwargs[channel_name] = int(round(kwargs[channel_name]))
    if kwargs[channel_name] < 1:
        warnings.warn("The number of channels is too small. Setting it to 1.")
        kwargs[channel_name] = 1

    equi_block = type_equi_block(**kwargs)
    if fix_params_mode in ["heuristic", "no"]:
        return equi_block

    param_normal_block = get_param_count(normal_block)
    if param_normal_block == 0:
        return equi_block

    equi_block = binary_search_fixed_params(type_equi_block, param_normal_block, 
                                            channel_name, max_iterations, **kwargs)
    return equi_block


def binary_search_fixed_params(type_equi_block, param_normal_block, channel_name, max_iterations, **kwargs):
    """
    Perform a binary search to find the optimal number of channels for the equivariant block
    to have a parameter count similar to the normal convolutional block.
    """
    param_equi_block = get_param_count(type_equi_block(**kwargs))
    out_channels = kwargs[channel_name]

    # Initialize search range
    if param_equi_block > param_normal_block:
        lower_bound = max(out_channels - 400, 1)
        upper_bound = out_channels
    else:
        lower_bound = out_channels
        upper_bound = int(round(out_channels // 0.5) + 100)

    old_equi_param, old_equi_conv_block = None, None
    iteration = 0

    # Binary search
    while lower_bound <= upper_bound and iteration < max_iterations:
        kwargs[channel_name] = (lower_bound + upper_bound) // 2
        old_equi_param, old_equi_conv_block = param_equi_block, type_equi_block(**kwargs)
        param_equi_block = get_param_count(old_equi_conv_block)

        if abs(param_equi_block - param_normal_block) < 0.01:
            return old_equi_conv_block

        if param_equi_block < param_normal_block:
            lower_bound = kwargs[channel_name] + 1
        else:
            upper_bound = kwargs[channel_name] - 1

        iteration += 1

    # Return closest channel size if no solution found
    if old_equi_param is not None and abs(old_equi_param - param_normal_block) < abs(param_equi_block - param_normal_block):
        return old_equi_conv_block
    return type_equi_block(**kwargs)


def get_param_count(model_name, in_mb=False, verbose=False):
        """Get the number of parameters of a given model.
        Args:
            params (tensor): Input tensor.
        Returns:
            Number of parameters of a given model.
        """
        if in_mb:
            param_size = 0
            for param in model_name.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model_name.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / 1024**2
            if verbose:
                print(f'Total size: {size_all_mb:.2f} MB')
            return size_all_mb
        else:
            total_params = sum(p.numel() for p in model_name.parameters()) / 1e6
            if verbose:
                print(f'Total params: {total_params}')
            return total_params


def get_gspace_from_name(group, rotation):
        """Get group space for a given group and rotation.
        Args:
            group (str): Group name.
            rotation (int): Rotation.
        Returns:
            gspace: Group space.
        """
        if group == "cyclic":
            gspace = rot2dOnR2(rotation)
        elif group == "dihedral":
            gspace = flipRot2dOnR2(rotation)
        elif group == "orthogonal":
            gspace = flipRot2dOnR2(-1)
        else:
            raise ValueError(
                f'Group "{group}" is not know. Available groups: [cyclic, dihedral, orthogonal]'
            )
        return gspace

def get_gspace_from_id(id):
        """Get group space from id.
        Args:
            id (tuple): Group id.
        Returns:
            gspace: Group space.
        """
        if isinstance(id, Tuple):
            reflection, rotation = id
        elif isinstance(id, int):
            reflection, rotation = -1, id
        else:
            raise ValueError(
                f'Group id "{id}" is not know.'
            )

        if reflection is None:
            # cyclic
            gspace = rot2dOnR2(rotation)
        elif reflection >= 0:
            # dihedral
            gspace = flipRot2dOnR2(rotation)
        else:
            raise ValueError(
                f'Group id "{id}" is not know.'
            )
        return gspace


if __name__ == "__main__":
    gspace = flipRot2dOnR2(4)
    print(gspace.fibergroup.regular_representation)