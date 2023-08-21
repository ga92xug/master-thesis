from typing import List
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import re
import pandas as pd
import sys
import os
# issue with https://github.com/pytorch/pytorch/issues/37377
os.environ["MKL_THREADING_LAYER"]="GNU"
sys.path.append(f"{os.getcwd()}")
from NAS.util import encode_parameters, convert_dict_to_hydra_string
from experiment.run_files.run_command import run_command


def get_adjusted_dict(
        cfg: DictConfig,
        dataset_name: str, 
        strategy_name: str,
        strategy_dict: dict,
        adjust_list: List[str],
    ):
    
    replacement_group = cfg.map_strategies_2_replacement_groups[dataset_name]
    replacement_group_strategy = "strategy_pure_" + replacement_group

    if replacement_group != strategy_name and len(adjust_list) > 0:
        pure_strategies_df = pd.read_csv('../../dev1/scaling-laws-ecnn/NAS/data/common_best_architecture/pure_strategies.csv', index_col=0)

        replacement_group_strategy_row = pure_strategies_df.loc[[replacement_group_strategy]]
        replacement_group_strategy_dict = replacement_group_strategy_row.squeeze().to_dict()

        adjusted = ""
        for to_adjust in adjust_list:
            
            adjusted += "_" + to_adjust

            if to_adjust == "group":
                # we have to adjust the reflection and the group
                to_adjust = ["_reflection", "_group"]
            

            # we replace the (X_reflection, X_group) of strategy with the one of dataset
            for key in strategy_dict.keys():
                if isinstance(to_adjust, list):
                    for adjust in to_adjust:
                        if adjust in key:
                            strategy_dict[key] = replacement_group_strategy_dict[key]
                elif isinstance(to_adjust, str):
                    if to_adjust in key:
                        strategy_dict[key] = replacement_group_strategy_dict[key]
                else:
                    raise ValueError("adjust_list should be a list of strings or a string")

        adjusted += "_adjusted"

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
        f'wandb.tags=[SE_test]',
        f'model.dropout_rate={strategy_dict["-1_dropout_rate"]}',
        f'model.eq_expand_ratio={strategy_dict["-1_expand_ratio"]}',
        f'+model.replacement_group_strategy={replacement_group_strategy}',
        f'+model.blocks_args_dict={convert_dict_to_hydra_string(strategy_dict)}',
        f'wandb.give_name=strategy_{strategy_name}{adjusted}',
    ]

    if dataset == "cifar10_rot":
        training_name = "cifar10"
        args.append('training.dataset.name=cifar10_rot')
    elif "mnist" in dataset:
        training_name = "mnist"
        args.append(f"training.dataset.name={dataset}")
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
            #print("Run successful")
            break
            #batch_size = get_batch_size_from_args(args)  # Replace with the actual way to get batch size from args
        except Exception as e:
            print(f"Exception: {e}")
            if dataset in ["galaxy10", "isic2019", "stl10"]:
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
        cfg: DictConfig, 
        datasets: list,
        strategies_df: pd.DataFrame,
        adjust: str, 
    ):    
    global_args = ["model=eq_nasnet"]

    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for strategy_row in strategies_df.iterrows():
            strategy_name = "_".join(strategy_row[0].split('_')[1:])
            
            if runs_to_skip(dataset, strategy_name, adjust):
                continue
            
            strategy_dict = strategy_row[1].to_dict()
            strategy_dict, replacement_group_strategy, adjusted = get_adjusted_dict(cfg, dataset, strategy_name, strategy_dict, adjust)
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

def strategy_on_datasets(
        cfg: DictConfig,
        datasets: list,
        strategy_name: str,
        strategy_dict: dict,
        adjust_list: List[str], 
    ):    
    global_args = ["model=eq_nasnet"]

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        
        if runs_to_skip(dataset, strategy_name, adjust_list):
            continue
        
        strategy_dict, replacement_group_strategy, adjusted = get_adjusted_dict(cfg, dataset, strategy_name, strategy_dict, adjust_list)
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
        adjust: List[str],
    ):
    ############################################################################
    # Always skip these 
    if dataset in strategy_name and len(adjust) > 0:
        print(f"Skipping Strategy: {strategy_name} on {dataset}, since same with no group adjustment")
        return True
    
    ############################################################################
    # Skip these temporarily
    if "fix" in strategy_name and len(adjust) > 1:
        return False
    
    if "pure" in strategy_name and len(adjust) <= 1:
        return False

    #if dataset == "galaxy10":
    #    if (strategy_name == "strategy_pure_galaxy10" and len(adjust) == 1) or \
    #        ("fix" in strategy_name and len(adjust) == 2):
    #    #print(f"Skipping Strategy: {strategy_name} on {dataset}, since already ran")
    #        return False

    #if "cifar" in dataset:
    #    if "fix" in strategy_name and len(adjust) == 3:
    #        return False
    
    return True


def main():
    # datasets = ["cifar10", "cifar10_rot", "galaxy10", "mnist12k", "mnist_rot"] #  "ISIC_2019"
    # datasets = ["mnist12k", "ISIC_2019"]
    strategies_df = pd.read_csv('../../dev1/scaling-laws-ecnn/NAS/data/common_best_architecture/pure_strategies.csv', index_col=0)

    for strategy in strategies_df.index:
        for adjust in [["group", "out_channels"]]:
            if adjust:
                datasets = ["mnist12k"]
            else:
                datasets = ["mnist12k", "galaxy10"]
                # datasets = ["cifar10", "cifar10_rot", "mnist12k", "mnist_rot", "galaxy10"]

            strategy_on_datasets(datasets, strategies_df, adjust)


def iter_over_strategies(cfg: DictConfig):
    datasets = list(cfg.datasets) #  "ISIC_2019"
    pure_strategies_df = pd.read_csv('../../dev1/scaling-laws-ecnn/NAS/data/common_best_architecture/pure_strategies.csv', index_col=0)
    mixed_strategies_df = pd.read_csv('../../dev1/scaling-laws-ecnn/NAS/data/common_best_architecture/mixed_strategies.csv', index_col=0)
    strategies_df = pd.concat([pure_strategies_df, mixed_strategies_df])

    for strategy in cfg.strategies:
        print(f"\nStrategy: {strategy}")
        for adjust_list in cfg.adjust_lists:

            strategy_dict = strategies_df.loc[[strategy]].squeeze().to_dict()
            strategy_on_datasets(cfg, datasets, strategy, strategy_dict, adjust_list)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_run = wandb.init(
        project=cfg.wandb.project, config=wandb_config, \
        job_type="run_strategies", mode=cfg.wandb.mode, notes=cfg.wandb.notes, \
        tags=cfg.wandb.tags
    )
    wandb_run.log_code(".")
    wandb.finish()
    iter_over_strategies(cfg)


if __name__ == "__main__":
    main()