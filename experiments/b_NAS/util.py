import wandb
import os

def encode_parameters(params, choice_2_range_params : dict = {
        "group": [1, 2, 4, 8, 16],
    }):
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
        reflection = params['%d_reflection' % i]
        kernel_size = params['%d_kernel_size' % i]
        try:
            group = choice_2_range_params['group'][int(params['%d_group' % i])]
        except:
            group = "*"
        out_channels = params['%d_out_channels' % i]
        stride = params['%d_stride' % i]
        #print('stride', stride)

        if i == blocks - 1:
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

    #print('encoded_blocks', encoded_blocks)
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

def convert_dict_to_hydra_string(dictionary: dict):
    key_value_comma = ",".join([f'{key}:{value}' for key, value in dictionary.items()])
    return f'{{{key_value_comma}}}'

def convert_to_number(val):
    try:
        if '.' in val:
            return float(val)
        else:
            return int(val)
    except ValueError:
        return val

def init_wandb(run_id, cfg, wandb_config=None):
    if run_id is not None:
        # resume wandb run
        run = wandb.init(
            project=cfg.wandb.project, 
            entity=cfg.wandb.entity, 
            mode=cfg.wandb.mode,
            resume="allow",
            id=run_id,  # resume the run using the saved run ID
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project, 
            entity=cfg.wandb.entity, 
            mode=cfg.wandb.mode,
            config=wandb_config,
        )

    # wandb_logger = logging.getLogger('wandb')
    # wandb_logger.setLevel(logging.ERROR)  # Set to ERROR to suppress most console output
    run.log_code(".")
    return run

def get_largest_saved_version_ax_client(folder: str = None):
    if folder is None:
        global save_folder
        folder = save_folder
    # get the largest version
    version = -1
    result = None
    for file in os.listdir(folder):
        if file.startswith("ax_client_"):
            file_version = int(file.split("_")[-1].split(".")[0])
            if file_version > version:
                version = file_version
                result = file
        
        if file == "ax_client.json":
            result = file
            version = 0

    print("Largest version: ", version, result)
    return version, result