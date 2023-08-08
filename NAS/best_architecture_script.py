import numpy as np
import wandb
import pandas as pd

import random
random.seed(42)

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df

import sys
import os
# issue with https://github.com/pytorch/pytorch/issues/37377
os.environ["MKL_THREADING_LAYER"]="GNU"
sys.path.append(f"{os.getcwd()}")
sys.path.append('../experiment')
sys.path.append('../../NAS')
#from networks.eq_nasnet.eq_nasnet import EquivariantNASNet
from NAS.util import encode_parameters, convert_to_number
from experiment.run_files.run_command import run_command



galaxy_client = AxClient.load_from_json_file(filepath="NAS/data/galaxy10_2.2.3/ax_client.json")
cifar_client = AxClient.load_from_json_file(filepath="NAS/data/cifar10_2.2/ax_client.json")
mnist_rot_client = AxClient.load_from_json_file(filepath="NAS/data/mnist_rot_2.2/ax_client.json")

#pd.set_option('display.max_columns', None)
galaxy_df = exp_to_df(galaxy_client.experiment)
galaxy_df = galaxy_df.drop_duplicates(subset=['arm_name'], keep=False)
#galaxy_df.sort_values(by=["valid_acc"], ascending=False).head(20)


mnist_rot_df = exp_to_df(mnist_rot_client.experiment)
mnist_rot_df = mnist_rot_df.drop_duplicates(subset=['arm_name'], keep=False)
#mnist_rot_df.sort_values(by=["valid_acc"], ascending=False).head(20)


cifar_df = exp_to_df(cifar_client.experiment)
cifar_df = cifar_df.drop_duplicates(subset=['arm_name'], keep=False)
#cifar_df.sort_values(by=["valid_acc"], ascending=False).head(20)


def compute_weighted_average(top_n, valid_acc_weight=1):
    weights = valid_acc_weight * top_n['valid_acc'] + (1 - valid_acc_weight) * top_n['gflops']
    return weights / weights.sum()

def summarize_top_n(df, n=20, valid_acc_weight=1, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method', 'model_building_time', 'is_feasible']

    top_n = df.nlargest(n, 'valid_acc')
    weights = compute_weighted_average(top_n, valid_acc_weight)


    summary = {}
    for col in top_n.columns:
        if col not in exclude_columns:
            if pd.api.types.is_numeric_dtype(top_n[col]):
                summary[col] = (top_n[col] * weights).sum()
            else:
                counts = top_n.groupby(col).apply(lambda x: (x['valid_acc'] * weights.loc[x.index]).sum())
                summary_string = ""
                for value, percentage in (counts / counts.sum() * 100).items():
                    summary_string += f"{value}={int(percentage)}%; "
                summary[col] = summary_string

    summary_df = pd.DataFrame(summary, index=['top_n'])
    return summary_df

def plot_impact_of_weighted_average(df, n=20):
    weights = [i/10 for i in range(11)]
    results = []

    for weight in weights:
        summary_df = summarize_top_n(df, n=n, valid_acc_weight=weight)
        results.append(summary_df.loc['top_n'].tolist())

    results_df = pd.DataFrame(results, columns=summary_df.columns, index=weights)
    results_df.plot(figsize=(10, 6), title="Impact of Weighted Averaging")
    plt.xlabel("valid_acc_weight")
    plt.ylabel("Value")
    plt.show()

# Sample usage
n = 20
summary_df = summarize_top_n(mnist_rot_df, n=n, valid_acc_weight=0.8)  # 0.8 is the weight for valid_acc in the averaging
summary_df


# List of DataFrames
dataframes = [mnist_rot_df, cifar_df, galaxy_df]  # Replace these with your actual DataFrames
dataset_names = ['mnist_rot', 'cifar', 'galaxy']
valid_acc_weight = 0.9
n = 20

# Create a list of summary DataFrames
summary_dfs = [summarize_top_n(df, n=n, valid_acc_weight=valid_acc_weight) for df in dataframes]

# Set the dataset name as the index for each summary DataFrame
for name, summary_df in zip(dataset_names, summary_dfs):
    summary_df.index = [name]

# Concatenate the summary DataFrames into one
final_summary_df = pd.concat(summary_dfs)

# Now you can print or use final_summary_df
final_summary_df



columns = final_summary_df.columns.tolist()
columns.remove("gflops")
columns.remove("valid_acc")
columns
config_str = "6	0.0	0	4.0	2	5.0	2.0	0.0	4.0	1.0	mbconv,conv	3.0	0.0,0.5	1.0,2.5	conv	1.0	0.0	*	2.0	conv	3	0.25,0.5 	3.8	no	1.0	-1	*	2	mbconv	5	0.0,0.5	1.0	conv	2	-1	*	1.0	3.0,5.0	2"
config_list = []
temp_item = ""



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
    "cifar10": [4, 4, 2, 1, 1], # last might also be 0
    "mnist_rot": [4, 4, 4, 4, 4],
    "galaxy10": [4, 2, 2, 2, 2], # last might also be 1
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
num_runs = 5

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
        config = fill_group_based_on_dataset(random_config.copy(), dataset)
        only_group_info = {k: v for k, v in config.items() if k.split("_")[1] == "group"}

        blocks_args = encode_parameters(config, choice_2_range_params)
        print(f'{i+1} for {dataset}: {blocks_args}')

        tag = "_".join([f"{key}_{value}" for key, value in random_config.items()])

        args = [
            f"training={dataset}-training",
            f"wandb.tags=[eqnas_result_test_random]",
            f"wandb.notes={tag}",
            f"model.blocks_args={blocks_args}",
        ]
        
        if dataset == "cifar10" and i == 0:
            # already done
            continue

        run_command(args, global_args, test=False, path="experiment/")
