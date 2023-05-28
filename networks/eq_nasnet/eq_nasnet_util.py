import re
import math
import collections
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('../networks') # add parent directory


CHANNELS_CONSTANT = 1

################################################################################
# Help functions for model architecture
################################################################################


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
        'reflection', 'group', 'kernel_size', 'stride', 'out_channels',
        'num_layers', 'conv_op', 'se_ratio', 'skip'])
# Set GlobalParams and BlockArgs's defaults
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def eq_round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.
    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new_filters: New filters number after calculating.
    """
    print("filters: ", filters)
    multiplier = width_coefficient
    if not multiplier:
        return int(round(filters))
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = depth_divisor
    min_depth = min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
         new_filters += divisor
    # new_filters /= rotation
    return int(round(new_filters))


def round_repeats(repeats, depth_coefficient):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

################################################################################
# Helper functions for loading model params
################################################################################

# BlockDecoder: A Class for encoding and decoding BlockArgs
# efficientnet_params: A function to query compound coefficient
# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet
# url_map and url_map_advprop: Dicts of url_map for pretrained weights
# load_pretrained_weights: A function to load pretrained weights

class BlockDecoder(object):
    """
        reflection,
        kernel_size,
        group,
        out_channels,
        stride,
        
        num_layers,
        conv_op,
        se_ratio,
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s1_e1_i32_o16_se0.25_noskip'.
        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(?<=[a-zA-Z])(?=[^a-zA-Z])', op)
            splits[1] = re.sub(r'-(?=\D)', '', splits[1])
            #print(op, splits)
            key, value = splits
            options[key] = value

        return BlockArgs(
            # all blocks have these params
            reflection=     int(options['r']),
            group=          int(options['g']),
            # 0 - k-1 blocks have these params 
            kernel_size=    int(options['k']) if 'k' in options else None,
            stride=         int(options['s']) if 's' in options else None,
            out_channels=   int(options['o']) if 'o' in options else None,
            # only 1 - k-1 middle blocks have these params
            num_layers=     int(options['n']) if 'n' in options else None,
            conv_op=        str(options['c']) if 'c' in options else None,
            se_ratio=       float(options['se']) if 'se' in options else None,
            skip=           str(options['sk']) if 'c' in options else None,

            # not used for now
            #expand_ratio=   int(options['e']),
            #input_filters=  int(options['i']),
            )

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.
        Args:
            block (namedtuple): A BlockArgs type argument.
        Returns:
            block_string: A String form of BlockArgs.
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d' % block.stride,
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'se%s' % block.se_ratio,
            block.skip,
        ]
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))

        BlockDecoder._check_valid_blocks_args(blocks_args)
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.
        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings
    

    @staticmethod
    def _check_valid_blocks_args(blocks_args):
        """Helper function for checking argument values in blocks_args.
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.
        """
        
        for i, block in enumerate(blocks_args):
            assert isinstance(block.out_channels, int) and block.out_channels > 0
            assert isinstance(block.kernel_size, int) and block.kernel_size > 0
            assert isinstance(block.stride, int) and block.stride > 0
            assert isinstance(block.group, int) and block.group > 0
            assert isinstance(block.reflection, int) and block.reflection in [-1,0]
            
            if i > 0:
                assert isinstance(block.num_layers, int) and block.num_layers > 0
                assert isinstance(block.conv_op, str) and block.conv_op in ["conv", "dconv", "mbconv"]
                assert isinstance(block.se_ratio, float) and 0 <= block.se_ratio <= 1
                assert isinstance(block.skip, str) and block.skip in ["identity", "no"]

                assert previous_block.reflection >= block.reflection
                assert previous_block.group >= block.group

            previous_block = block


if __name__ == "__main__":
    blocks_args = ['r0_k3_g8_o1_s2', 'r0_k3_g2_o1_s2_n1_c-conv_se0.25_sk-identity', 'r0_k3_g2_o1_s2_n1_c-mbconv_se0.25_sk-identity', 'r-1_k3_g1_o1_s2_n2_c-mbconv_se0.25_sk-no']
    blocks_args = BlockDecoder.decode(blocks_args)