######################################################################
# Encode the parameters of the search space into block_args structure for Eq_NASNet

def encode_parameters(params):
    """
    Encodes the parameters into a string representation.

    Args:
        params (dict): A dictionary containing the parameters.

    Returns:
        str: The encoded string representation of the parameters.
    """
    blocks = params['i_num_layers']  # Number of blocks
    encoded_blocks = []
    
    for i in range(blocks):
        block_args = [
            'r%d' % params['%d_reflection' % i],
            'k%d' % params['%d_kernel_size' % i],
            'i%d' % params['%d_group' % i],
            'o%d' % params['%d_out_channels' % i],
        ]

        if i > 0:
            block_args.extend([
                'n%d' % params['%d_num_layers' % i],
                'c%s' % params['%d_conv_op' % i],
                'se%s' % params['%d_se_ratio' % i],
                params['%d_skip_op' % i],
            ])

        encoded_blocks.append('_'.join(block_args))

    return encoded_blocks


def encode_single_block_parameters(params, block_number):
    """
    Encodes the parameters of a single block into a string representation.

    Args:
        params (dict): A dictionary containing the parameters.
        block_number (int): The number of the block.

    Returns:
        str: The encoded string representation of the block parameters.
    """
    block_args = [
        'r%d' % params['%d_reflection' % block_number],
        'k%d' % params['%d_kernel_size' % block_number],
        'i%d' % params['%d_group' % block_number],
        'o%d' % params['%d_out_channels' % block_number],
    ]

    if block_number > 0:
        block_args.extend([
            'n%d' % params['%d_num_layers' % block_number],
            'c%s' % params['%d_conv_op' % block_number],
            'se%s' % params['%d_se_ratio' % block_number],
            params['%d_skip_op' % block_number],
        ])

    return '_'.join(block_args)


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