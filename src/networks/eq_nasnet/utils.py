import math
from typing import Dict, List, Tuple, Union
import sys
import os
sys.path.append(f"{os.getcwd()}")
from equivariant.nn.gspace import GSpace
from src.networks.eq_nasnet.block_args import BlockArgs, BlockArgsList


def pool_like(out_channels: int, num_classes: int, image_size) -> Tuple[int, int]:
    if out_channels >= num_classes:
        # we can fully pool over spatial dimensions
        return 1, out_channels
    
    pooling_size = int(math.ceil(math.sqrt(num_classes / out_channels)))
    assert pooling_size > 1, "Pooling size must be greater than 1"
    assert pooling_size <= image_size, "Pooling size must be less than or equal to image size"
    return pooling_size, out_channels * pooling_size * pooling_size

################################################################################
# utils for naming Eq-NasNet
################################################################################

def set_eq_nasnet_name(
        pre_trained: bool,
        gspace: GSpace,
        blocks_args_list: BlockArgsList,
        depth_coefficient: float,
        width_coefficient: float,
        image_size: int,
        dropout_rate: float,            
    ) -> Dict[str, str]:
    pre_trained = "-pre" if pre_trained else ""
    model_name = f"Eq-NasNet{pre_trained}-{gspace.fibergroup}-b{len(blocks_args_list)-2}-d{depth_coefficient}-w{width_coefficient}-r{image_size}-drop{dropout_rate:.2f}"
    scaling_name = get_scaling_name(
        blocks_args_list=blocks_args_list,
        width_coefficient=width_coefficient,
        resolution=image_size,    
    )
    return {
        "model_name": model_name,
        "scaling_name": scaling_name,
    }

def get_scaling_name(
        blocks_args_list: BlockArgsList,
        width_coefficient: float,
        resolution: int,
    ) -> str:
    """
    Returns the scaling name for the given config.
    """
    # stem and head are not counted as blocks
    num_blocks = len(blocks_args_list) - 2

    # depth
    num_layers_blocks = ""
    for block_args in blocks_args_list[1:-1]:
        num_layers_block = block_args.num_layers
        num_layers_blocks += "-" + str(num_layers_block)
    # remove first "-"
    num_layer_blocks = num_layers_blocks[1:]

    # width
    width_coefficient = float_to_int_if_possible(width_coefficient)

    scaling_name = f"b{num_blocks}_d-{num_layer_blocks}_w{width_coefficient}_r{resolution}"
    return scaling_name

def float_to_int_if_possible(x):
    if x == int(x):
        return int(x)
    return x