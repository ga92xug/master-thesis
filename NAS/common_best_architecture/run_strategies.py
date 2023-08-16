import re
from turtle import st
import pandas as pd
import sys
import os
# issue with https://github.com/pytorch/pytorch/issues/37377
os.environ["MKL_THREADING_LAYER"]="GNU"
sys.path.append(f"{os.getcwd()}")
from NAS.util import encode_parameters, convert_dict_to_hydra_string
from experiment.run_files.run_command import run_command


def get_adjusted_dict(
        dataset_name: str, 
        strategies_df: pd.DataFrame,
        strategy_name: str,
        strategy_dict: dict,
        adjust_group: bool,
    ):
    map_strategies_2_replacement_groups = {
        "cifar10": "cifar10",
        "cifar10_rot": "mnist_rot",
        "galaxy10": "galaxy10",
        "mnist12k": "cifar10",
        "mnist_rot": "mnist_rot",
        "ISIC_2019": "galaxy10"
    }
    replacement_group = map_strategies_2_replacement_groups[dataset_name]
    replacement_group_strategy = "strategy_" + replacement_group

    if replacement_group != strategy_name and adjust_group:
        adjusted = "_group_adjusted"

        replacement_group_strategy_row = strategies_df.loc[[replacement_group_strategy]]
        replacement_group_strategy_dict = replacement_group_strategy_row.squeeze().to_dict()

        # we replace the (X_reflection, X_group) of strategy with the one of dataset
        for key in strategy_dict.keys():
            if '_reflection' in key or '_group' in key:
                strategy_dict[key] = replacement_group_strategy_dict[key]

    else:
        adjusted = ""
        replacement_group_strategy = ""

    return strategy_dict, replacement_group_strategy, adjusted

def get_training_args(
        dataset: str,
        strategy_name: str,
        adjusted: str,
        replacement_group_strategy: str,
        strategy_dict: dict,
    ):

    # Create a new run
    args = [
        f'wandb.tags=[eqnasnet,find_best,pure_strategy]',
        f'model.dropout_rate={strategy_dict["-1_dropout_rate"]}',
        f'model.eq_expand_ratio={strategy_dict["-1_expand_ratio"]}',
        f'+model.replacement_group_strategy={replacement_group_strategy}',
        f'+model.blocks_args_dict={convert_dict_to_hydra_string(strategy_dict)}',
        f'wandb.give_name=pure_strategy_{strategy_name}{adjusted}',
    ]

    if dataset == "cifar10_rot":
        training_name = "cifar10"
        args.append('training.dataset.name=cifar10_rot')
    elif dataset == "mnist":
        training_name = "mnist_rot"
        args.append("training.dataset.name=mnist")
    else:
        training_name = dataset

    args.append(f'training={training_name}-training')

    return args

def start_run(
        args: list,
        global_args: list,
        dataset: str,
        max_retries: int = 4,
    ):
    retry_count = 0

    batch_size = 128
    accumulate = 2

    while retry_count < max_retries:
        try:
            run_command(args, global_args, test=False, path="experiment/")
            break
            #batch_size = get_batch_size_from_args(args)  # Replace with the actual way to get batch size from args
        except Exception as e:
            print(f"Exception: {e}")
            if dataset in ["galaxy10", "ISIC_2019"]:
                batch_size = batch_size // 2
                accumulate = accumulate * 2
                
                args.append(f"training.dataset.batch_size={batch_size}")
                args.append(f"training.dataset.eval_batch_size={batch_size}")
                args.append(f"training.accumulate={accumulate}")
                retry_count += 1
            else:
                # We don't retry for other datasets
                break

    if retry_count == max_retries:
        print("Maximum retries reached. Giving up.")


def strategies_on_datasets(
       datasets: list,
       strategies_df: pd.DataFrame,
       adjust_group: bool, 
    ):    
    global_args = ["model=eq_nasnet"]

    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for strategy_row in strategies_df.iterrows():
            strategy_name = "_".join(strategy_row[0].split('_')[1:])
            
            if runs_to_skip(dataset, strategy_name, adjust_group):
                continue
            
            strategy_dict = strategy_row[1].to_dict()
            strategy_dict, replacement_group_strategy, adjusted = get_adjusted_dict(dataset, strategies_df, strategy_name, strategy_dict, adjust_group)
            if replacement_group_strategy != "":
                tmp_string = f" replaced: {replacement_group_strategy}"
            else:
                tmp_string = ""
            print(f"Strategy: {strategy_name}{adjusted}{tmp_string}")
            
            args = get_training_args(
                dataset,
                strategy_name,
                adjusted,
                replacement_group_strategy,
                strategy_dict,
            )
            
            #print(f"Strategy dict: {strategy_dict}")
            #print(f"Args: {args}")

            start_run(args, global_args, dataset)

def runs_to_skip(
        dataset: str,
        strategy_name: str,
        adjust_group: bool,
    ):
    ############################################################################
    # Always skip these 
    if dataset == strategy_name and not adjust_group:
        print(f"Skipping Strategy: {strategy_name} on {dataset}, since same with no group adjustment")
        return True
    
    ############################################################################
    # Skip these temporarily
    if (dataset == "cifar10_rot" and strategy_name == "mnist_rot" and adjust_group) or \
        (dataset == "cifar10_rot" and strategy_name == "mnist_rot" and not adjust_group) or \
        print(f"Skipping Strategy: {strategy_name} on {dataset}, since already ran"):
        return True
    
    return False



def main():
    # datasets = ["cifar10", "cifar10_rot", "galaxy10", "mnist", "mnist_rot"] #  "ISIC_2019"
    # datasets = ["mnist12k", "ISIC_2019"]
    strategies_df = pd.read_csv('../../dev1/scaling-laws-ecnn/NAS/data/common_best_architecture/pure_strategies.csv', index_col=0)

    for adjust_group in [True, False]:
        if adjust_group:
            continue
        else:
            datasets = ["galaxy10"]
            # datasets = ["cifar10", "cifar10_rot", "mnist12k", "mnist_rot", "galaxy10"]

        strategies_on_datasets(datasets, strategies_df, adjust_group)



if __name__ == "__main__":
    main()