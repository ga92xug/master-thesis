import numpy as np
import wandb
import pandas as pd

import random
random.seed(0) # 42

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df

# issue with https://github.com/pytorch/pytorch/issues/37377
os.environ["MKL_THREADING_LAYER"]="GNU"
import sys
import os
sys.path.append(f"{os.getcwd()}")
from networks.eq_nasnet.util import encode_parameters
from experiments.b_NAS.util import convert_to_number
from experiments.run_command import run_command


columns = ['-1_expand_ratio', '-1_dropout_rate', '0_reflection', '0_group', '0_out_channels', '0_kernel_size', '0_stride', '1_reflection', '1_group', '1_num_layers', '1_conv_op', '1_kernel_size', '1_se_ratio', '1_out_channels', '1_skip_op', '1_stride', '2_reflection', '2_group', '2_num_layers', '2_conv_op', '2_kernel_size', '2_se_ratio', '2_out_channels', '2_skip_op', '2_stride', '3_reflection', '3_group', '3_num_layers', '3_conv_op', '3_kernel_size', '3_se_ratio', '3_out_channels', '3_skip_op', '3_stride', '4_reflection', '4_group', '4_out_channels', '4_kernel_size', '4_stride']
#strategy_1_0
#strategy_1_1 = "6	0.0	0	4.0	2	5.0	2.0	0.0	*	1.0	mbconv	3.0	0.0	1.0,2.5	conv	1.0	0.0	*	2.0	conv	3	0.25,0.5 	3.8	no	1.0	-1	*	2	mbconv	5	0.0,0.5	1.0	conv	2	-1	*	1.0	3.0,5.0	2"
strategy_2_0 = "4	0.000000	0.000000	4	1.630486	5.0	2.0	0.000000	*	1.000000	conv	3.0	0.000000	1.014946	conv,identity	1.000000	0.000000	*	2.000000	conv	3	0.0,0.25	3.983971	identity,no	1.0	-1.000000	*	2.000000	mbconv	5.000000	0.000000	3.944093	conv,identity	2	-1.000000	1,2	1.0	3.000000	1.000000"
config_str = strategy_2_0

config_list = []
temp_item = ""

for item in config_str.split():
    if ',' in item:
        sub_items = [convert_to_number(sub_item) for sub_item in item.split(',')]
        config_list.append(sub_items)
    else:
        config_list.append(convert_to_number(item))

        # Check if the item ends with a comma
        if item.endswith(','):
            temp_item = item[:-1]

print(config_list)
dict_config = dict(zip(columns, config_list))
dict_config


global_args = ["model=eq_nasnet"]

groups_for_datasets = {
    "cifar10": [4, 4, 2, 2, 0], # last might also be 0
    "mnist_rot": [4, 4, 4, 4, 4],
    "galaxy10": [4, 2, 2, 2, 2], # last might also be 1
    "isic2019": [4, 2, 2, 2, 2],
} 

choice_2_range_params = {
    "group": [1, 2, 4, 8, 16],
}

# Function to create a random configuration
def create_random_configuration(config):
    random_config = {}
    for key, value in config.items():
        if isinstance(value, list):
            random_choice = random.choice(value)
            random_config[key] = random_choice
        else:
            random_config[key] = value
            
    return random_config

def fill_group_based_on_dataset(config, dataset):
    for key, value in config.items():
        if value == "*":
            value = groups_for_datasets[dataset][int(key.split("_")[0])]
            config[key] = value
    return config

# Number of runs
num_runs = 3

# Keep track of random configurations
random_configurations = []

# Iterate for each run
for i in range(num_runs):
    while True:
        random_config = create_random_configuration(dict_config)
        if random_config not in random_configurations:
            random_configurations.append(random_config)
            break

    print(f'\nRandom Configuration {i+1}: {random_config}')

    for dataset in groups_for_datasets.keys():
        if i != 0 and dataset in ["mnist_rot", "cifar10"]:
            continue

        config = fill_group_based_on_dataset(random_config.copy(), dataset)
        only_group_info = {k: v for k, v in config.items() if k.split("_")[1] == "group"}

        assert False, "currently not working. Logic of encode params changed"
        blocks_args = encode_parameters(config, True, choice_2_range_params)
        print(f'{i+1} for {dataset}: {blocks_args}')

        tag = "_".join([f"{key}_{value}" for key, value in random_config.items()])

        args = [
            f"training={dataset}-training",
            f"wandb.tags=[eqnas_result_test_random]",
            f"wandb.notes={tag}",
            f"model.blocks_args={blocks_args}",
            f"model.dropout_rate={random_config['-1_dropout_rate']}",
            f"model.eq_expand_ratio={random_config['-1_expand_ratio']}",
        ]

        run_command(args, global_args, test=False, path="experiment/")
