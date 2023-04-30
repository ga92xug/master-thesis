import math

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

def iter_fix_param(type_equi_block, normal_block, fix_params, 
                   channel_name="out_channels", **kwargs):
    """
    TODO
    """
    equi_block = type_equi_block(**kwargs)
    if not fix_params:
        return equi_block
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
        upper_bound = int(round(out_channels // 0.7))
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
            print(f'Ratio for block: {last_ratio}')
            return equi_block
        if param_equi_block < param_normal_block:
            # prediction is too small
            lower_bound = kwargs[channel_name] + 1
            if lower_bound >= upper_bound:
                # we increase to upper bound slowly to avoid expensive search
                upper_bound += 20
                 
        else:
            upper_bound = kwargs[channel_name] - 1
                
    # if no solution found, return closest channel size
    if old_equi_param is not None:
        if abs(old_equi_param - param_normal_block) < abs(param_equi_block - param_normal_block):
            equi_block = old_equi_conv_block
        
    last_ratio = param_equi_block / param_normal_block
    print(f'Ratio for block: {last_ratio}')
    return equi_block

def get_param_count(model_name):
        """Get the number of parameters of a given model.
        Args:
            params (tensor): Input tensor.
        Returns:
            Number of parameters of a given model.
        """
        return sum(p.numel() for p in model_name.parameters() if p.requires_grad)