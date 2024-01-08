from typing import Dict

from src.networks.eq_nasnet.block_args import BlockArgs

def get_scaling_name(
        blocks_args: BlockArgs,
        width_coefficient: float,
        resolution: int,
    ) -> str:
    """
    Returns the scaling name for the given config.
    """
    # stem and head are not counted as blocks
    num_blocks = len(blocks_args) - 2

    # depth
    num_layers_blocks = ""
    for block_args in blocks_args[1:-1]:
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