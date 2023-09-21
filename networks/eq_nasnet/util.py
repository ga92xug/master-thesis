import re
import math
import collections
from typing import Tuple, Union
import torch
from torch import mul, nn
from torch.nn import functional as F
import sys
import os
sys.path.append(f"{os.getcwd()}")
from experiments.b_NAS.util import convert_to_number
from nn import FieldType

CHANNELS_CONSTANT = 1

################################################################################
# Help functions for model architecture
################################################################################


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
        'reflection', 'group', 'kernel_size', 'stride', 'out_channel',
        'num_layers', 'conv_op', 'se_ratio', 'skip'])

# Set GlobalParams and BlockArgs's defaults
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def get_channel_sizes(initial_channel_size, blocks_args, width_coefficient):
    list_out_channel = []
    old_channels = initial_channel_size
    for i, block_args in enumerate(blocks_args):
        # first in out_channels the increase factor is stored
        # we change that to the actual out_channels
        increase_factor = block_args.out_channel
        if i == 0:
            increase_factor *= width_coefficient

        out_channel = old_channels*increase_factor

        blocks_args[i] = block_args._replace(out_channel=out_channel)
        old_channels = out_channel
        list_out_channel.append(out_channel)
    
    print(f"list_out_channel: {list_out_channel}")
    return blocks_args

def get_fixed_out_channels2(
        in_type: Union[int, FieldType],
        increase_factor: float,
        new_rotation: int, 
        is_first: bool = False,
    ):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.
    Args:
        input_channels (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new_filters: New filters number after calculating.
    """
    if is_first:
        increased_channels = in_type * increase_factor
        out_channels = (increased_channels / new_rotation) * math.sqrt(new_rotation)

    elif isinstance(in_type, FieldType):
        old_rotation = in_type.gspace._sg_id[1]
        print(f"in_type: {in_type}, increase_factor: {increase_factor}, new_rotation: {new_rotation}, old_rotation: {old_rotation}")
        old_channels = len(in_type)
        old_initial_channels_size = (old_channels * old_rotation) / math.sqrt(old_rotation)
        print(f"old_initial_channels_size: {old_initial_channels_size}")
        old_initial_channels_size_increased = old_initial_channels_size * increase_factor

        out_channels = (old_initial_channels_size_increased / new_rotation) * math.sqrt(new_rotation)
    elif isinstance(in_type, int):
        out_channels = in_type * increase_factor
    else:
        raise ValueError(f"input_channels must be int or FieldType, got {type(in_type)}")

    out_channels = int(round(out_channels))
    print(f"out_channels: {out_channels}")
    return out_channels
    

def get_increase_factor(
        increase_factor: float,
        width_coefficient, 
    ):
    multiplier = width_coefficient
    if multiplier == 1:
        #print(f"increase_factor: {increase_factor}")
        return increase_factor
    else:
        return width_coefficient * increase_factor
        


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


def encode_parameters_old(params: dict, nas_encoded: bool = True, choice_2_range_params : dict = {
        "group": [1, 2, 4, 8, 16],
    }):
    """
    Encodes the parameters into a string representation. If group_encoded is True, the group parameter is encoded as a number from 0 to 4, otherwise it is encoded as a number from 1 to 16.

    Args:
        params (dict): A dictionary containing the parameters.
        choice_2_range_params (dict): A dictionary containing the discrete 
            choices for some of the params.

    Returns:
        str: The encoded string representation of the parameters.
    """
    # Number of blocks is determined by the highest numbered block in the keys of params

    num_blocks = max(int(key.split('_')[0]) for key in params.keys() if key.split('_')[0].isdigit()) + 1
    encoded_blocks = []
    
    for i in range(num_blocks):
        reflection = params['%d_reflection' % i]
        kernel_size = params['%d_kernel_size' % i]
        if nas_encoded:
            # group is encoded as a number from 0 to 4
            try:
                group = choice_2_range_params['group'][int(params['%d_group' % i])]
            except:
                group = "*"
        else:
            # group is encoded as a number from 1 to 16
            group = params['%d_group' % i]
        out_channels = params['%d_out_channels' % i]
        stride = params['%d_stride' % i]
        #print('stride', stride)

        if i == num_blocks - 1:
            # last block
            block_args = [
                'r%s' % reflection,
                'k%s' % kernel_size,
                'g%s' % group,
                'o%s' % out_channels,
            ]
        else:
            # start and middle blocks
            block_args = [
                'r%s' % reflection,
                'k%s' % kernel_size,
                'g%s' % group,
                'o%s' % out_channels,
                's%s' % stride,
            ]

            if i > 0:
                num_layers = params['%d_num_layers' % i]
                conv_op = params['%d_conv_op' % i]
                se_ratio = params['%d_se_ratio' % i]
                skip_op = params['%d_skip_op' % i]
                
                block_args.extend([
                    'n%s' % num_layers,
                    'c-%s' % conv_op,
                    'se%s' % se_ratio,
                    'sk-%s' % skip_op,
                ])

        encoded_blocks.append('_'.join(block_args))

    return encoded_blocks

def encode_parameters(
        params: dict, 
        nas_encoded: bool = True, 
        choice_2_range_params : dict = {"group": [1, 2, 4, 8, 16]},
        nas_key_2_eq_nasnet_key: dict = {
            # the names changed slightly
            "skip_op": "skip",
            "out_channels": "out_channel",  
            # the names that did not change
            "reflection": "reflection",
            "group": "group",
            "num_layers": "num_layers",
            "conv_op": "conv_op",
            "kernel_size": "kernel_size",
            "se_ratio": "se_ratio",
            "stride": "stride",

        },
    ):
    """
    Encodes the parameters into a dict representation. If group_encoded is True, the group parameter is encoded as a number from 0 to 4, otherwise it is encoded as a number from 1 to 16.

    Args:
        params (dict): A dictionary containing the parameters.
        choice_2_range_params (dict): A dictionary containing the discrete 
            choices for some of the params.

    Returns:
        dict: The encoded dict representation of the parameters.
    """
    # Number of blocks is determined by the highest numbered block in the keys of params

    num_blocks = max(int(key.split('_')[0]) for key in params.keys() if key.split('_')[0].isdigit()) + 1
    
    blocks = {i: {} for i in range(num_blocks)}

    for key, value in params.items():
        block_index, param_name = key.split("_")[0], "_".join(key.split("_")[1:])

        if param_name in nas_key_2_eq_nasnet_key:
            # convert the param name to the nasnet equivalent 
            # the names changed slightly
            param_name = nas_key_2_eq_nasnet_key[param_name]
        else:
            print(f"key {key} not found in nas_key_2_eq_nasnet_key")
            continue        

        if block_index.isdigit():
            block_index = int(block_index)
            if param_name == "group" and nas_encoded:
                # group is encoded as a number from 0 to 4 
                value = choice_2_range_params['group'][int(value)]
                
            blocks[block_index][param_name] = value
        else:
            raise ValueError(f"block_index should be a number, got {block_index}, type: {type(block_index)}, key: {key}, value: {value}")

    return blocks

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
        out_channel,
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
            #print(op)
            splits = re.split(r'(?<=[a-zA-Z])(?=[^a-zA-Z])', op)
            splits[1] = re.sub(r'-(?=\D)', '', splits[1])
            #print(op, splits)
            key, value = splits
            options[key] = convert_to_number(value)

            

        return BlockArgs(
            # all blocks have these params
            reflection=                 int(options['r']),
            group=                      int(options['g']),
            # 0 - k-1 blocks have these params 
            kernel_size=                int(options['k']) if 'k' in options else None,
            stride=                     int(options['s']) if 's' in options else None,
            out_channel=                float(options['o']) if 'o' in options else None,
            # only 1 - k-1 middle blocks have these params
            num_layers=                 int(options['n']) if 'n' in options else None,
            conv_op=                    str(options['c']) if 'c' in options else None,
            se_ratio=                   float(options['se']) if 'se' in options else None,
            skip=                       str(options['sk']) if 'c' in options else None,

            # not used for now
            #expand_ratio=   int(options['e']),
            #input_filters=  int(options['i']),
            )

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
        block_strings = encode_parameters(blocks_args)
        return block_strings
    

    @staticmethod
    def _check_valid_blocks_args(blocks_args):
        """Helper function for checking argument values in blocks_args.
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.
        """
        
        for i, block in enumerate(blocks_args):
            #print(i, block)
            assert block.out_channel > 0
            assert isinstance(block.kernel_size, int) and block.kernel_size >= 0
            assert isinstance(block.group, int) and block.group >= 0
            assert isinstance(block.reflection, int) and block.reflection in [-1,0]

            if i != len(blocks_args) - 1:
                # The last block has no stride
                assert isinstance(block.stride, int) and block.stride > 0
            
            if i > 0 and i < len(blocks_args) - 1:
                assert isinstance(block.num_layers, int) and block.num_layers > 0
                assert isinstance(block.conv_op, str) and block.conv_op in ["conv", "dconv", "mbconv"]
                assert isinstance(block.se_ratio, float) and 0 <= block.se_ratio <= 1
                assert isinstance(block.skip, str) and block.skip in ["identity", "no", "conv"], f"block.skip: {block.skip}, type: {type(block.skip)}"

                assert previous_block.reflection >= block.reflection
                assert previous_block.group >= block.group

            previous_block = block

    @staticmethod
    def legacy_code():
        """
        #blocks_args_dict = dict(kwargs.get("blocks_args_new", None))
                
        # flatten dict by prefixing the keys
        #blocks_args_dict = flatten_dict(blocks_args_dict)
        print("blocks_args_new: ", blocks_args_dict, type(blocks_args_dict))


        blocks_args_config = kwargs.get("blocks_args_dict", None)
        print("blocks_args_config: ", blocks_args_config)

        # compare 2 dicts to see if they are the same, and print the differences
        #compare_dicts(blocks_args_dict, blocks_args_config)

        print("blocks_args_config: ", blocks_args_config)
        if blocks_args_config is not None:
            # passed the config as a dict, ignore the default blocks_args
            # the encoder is informed about the conversion of the group
            blocks_args = encode_parameters(blocks_args_config, nas_encoded=False)
        
        blocks_args = BlockDecoder.decode(blocks_args)
        print("blocks_args: ", blocks_args)

        # update the config for wandb
        model_description = {}
        for i, block_args in enumerate(blocks_args):
            model_description[f"l{i}"] = block_args._asdict()
        
        try:
            pass
            #wandb.config.update({"model_description": model_description})
        except:
            # if wandb is not initialized during NAS
            pass
        """


if __name__ == "__main__":
    blocks_args = ['r0_k3_g8_o1_s2', 'r0_k3_g2_o1_s2_n1_c-conv_se0.25_sk-identity', 'r0_k3_g2_o1_s2_n1_c-mbconv_se0.25_sk-identity', 'r-1_k3_g1_o1_s2_n2_c-mbconv_se0.25_sk-no']
    blocks_args = BlockDecoder.decode(blocks_args)