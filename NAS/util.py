######################################################################
# Encode the parameters of the search space into block_args structure for Eq_NASNet

def encode_parameters(params, choice_2_range_params, strides):
    """
    Encodes the parameters into a string representation.

    Args:
        params (dict): A dictionary containing the parameters.
        choice_2_range_params (dict): A dictionary containing the discrete 
            choices for some of the params.

    Returns:
        str: The encoded string representation of the parameters.
    """
    # Number of blocks is determined by the highest numbered block in the keys of params
    blocks = max(int(key.split('_')[0]) for key in params.keys() if key.split('_')[0].isdigit()) + 1
    encoded_blocks = []
    
    for i in range(blocks):
        # For each parameter that is in the choice_2_range_params, convert it back to its original value
        reflection = params['%d_reflection' % i]
        kernel_size = choice_2_range_params['kernel_size'][params['%d_kernel_size' % i]]
        group = choice_2_range_params['group'][params['%d_group' % i]]
        out_channels = choice_2_range_params['out_channels'][params['%d_out_channels' % i]]

        if i == blocks - 1:
            # last block
            block_args = [
                'r%d' % reflection,
                'k%d' % kernel_size,
                'g%d' % group,
                'o%d' % out_channels,
            ]
        else:
            # start and middle blocks
            block_args = [
                'r%d' % reflection,
                'k%d' % kernel_size,
                'g%d' % group,
                'o%d' % out_channels,
                's%d' % strides[i],
            ]

            if i > 0:
                num_layers = params['%d_num_layers' % i]
                conv_op = params['%d_conv_op' % i]
                se_ratio = choice_2_range_params['se_ratio'][params['%d_se_ratio' % i]]
                skip_op = params['%d_skip_op' % i]
                
                block_args.extend([
                    'n%d' % num_layers,
                    'c-%s' % conv_op,
                    'se%s' % se_ratio,
                    'sk-%s' % skip_op,
                ])

        encoded_blocks.append('_'.join(block_args))

    
    return encoded_blocks


def decode_single_block_parameters(encoded_params, block_number):
    parts = encoded_params.split('_')
    params = {
        f'{block_number}_reflection': int(parts[0][1:]),
        f'{block_number}_kernel_size': int(parts[1][1:]),
        f'{block_number}_group': int(parts[2][1:]),
        f'{block_number}_out_channels': int(parts[3][1:]),
    }

    if block_number > 0:
        params.update({
            f'{block_number}_num_layers': int(parts[4][1:]),
            f'{block_number}_conv_op': parts[5][1:],
            f'{block_number}_se_ratio': float(parts[6][2:]),
            f'{block_number}_skip_op': parts[7],
        })

    return params