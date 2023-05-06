from copy import deepcopy
import math
import warnings
import numpy as np
import sys
import torch
sys.path.append('../scaling-laws-ecnn') # add parent directory

from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
)

CHANNELS_CONSTANT = 1

def cuda_memory_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"Allocated: {r / 1024 ** 3:.1f} GB")
    print(f"Allocated:    {a / 1024 ** 3:.1f} GB")
    print(f"Free:         {f / 1024 ** 3:.1f} GB")

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

def get_fixed_params(type_equi_block, fix_params_mode, normal_block=None, gspace=None,
                   channel_name="out_channels", **kwargs):
    """
    TODO
    """
    N = gspace.fibergroup.order()
    kwargs["out_channels"] = int(kwargs["out_channels"] / N)
    if fix_params_mode in ["heuristic", "all"]:
        kwargs["out_channels"] = int(kwargs["out_channels"] * math.sqrt(N * CHANNELS_CONSTANT))

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
            #print(f'Ratio for block: {last_ratio:.3f}')
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
    # print(f'Ratio for block: {last_ratio:.3f}')
    return equi_block

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
            total_params = sum(p.numel() for p in model_name.parameters())
            if verbose:
                print(f'Total params: {total_params}')
            return total_params



def get_gspace(group, rotation):
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

def calculate_fixed_params(num_c, gspace, restrict):
    CHANNELS_CONSTANT = 1
    # deepcopy to avoid changing the original list
    num_channels = deepcopy(num_c)
    diff = len(num_channels) - len(restrict)

    current_order = gspace.fibergroup.order()
    for l in range(len(num_channels)):
        
        if l >= diff and restrict[l-diff] is not None:
            if restrict[l-diff] == "halved":
                current_order = current_order // 2
                num_channels[l] = int(num_channels[l] * 2)
            elif restrict[l-diff] == "reflection":
                pass
            elif restrict[l-diff] == "invariant":
                current_order = 1
            
        num_channels[l] *= math.sqrt(current_order * CHANNELS_CONSTANT)

    
    return np.array(num_channels).astype(int)


if __name__ == "__main__":
    rotation = 4
    gspace = get_gspace("cyclic", rotation)

    num_c = np.array([32, 16, 24, 32, 64, 96, 160, 320, 1280])
    restrict = ["invariant", None]
    out = calculate_fixed_params(num_c, gspace, restrict)
    print(num_c)
    print(out)