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
from experiments.util import convert_dict_to_hydra_string
from networks.eq_nasnet.util import encode_parameters
from experiments.run_command import run_command

def get_replacement_dict(
        data_dir: str,
        replacement_name: str,    
    ):
    # read in the pure strategies
    pure_strategies_df = pd.read_csv(data_dir + 'pure_strategies.csv', index_col=0)
    replacement_row = pure_strategies_df.loc[["pure_" + replacement_name]]
    replacement_dict = replacement_row.squeeze().to_dict()

    replacement_dict = encode_parameters(replacement_dict)
    return replacement_dict

def remove_keys_from_dict(
        replacement_dict: dict,
        adjust_list: List[str],
    ):
    keys_to_remove = []
    for k1, v1 in replacement_dict.items():
        keys_to_remove_inner = []
        for k2, v2 in v1.items():
            if k2 not in adjust_list:
                keys_to_remove_inner.append(k2)

        for k2 in keys_to_remove_inner:
            del replacement_dict[k1][k2]

        if len(replacement_dict[k1]) == 0:
            keys_to_remove.append(k1)

    for k in keys_to_remove:
        del replacement_dict[k]

def get_adjusted_dict(
        map_strategies_2_replacement_groups: dict,
        data_dir: str,
        dataset_name: str, 
        strategy_name: str,
        adjust_list: List[str],
    ):
    if "group" in adjust_list:
        adjust_list.append("reflection")
    
    replacement_name = map_strategies_2_replacement_groups[dataset_name]

    change_logic = ""
    replacement_dict = {}
    if replacement_name != strategy_name and len(adjust_list) > 0:
        change_logic += "from:_" + replacement_name + "_with:"
        replacement_dict = get_replacement_dict(
            data_dir, 
            replacement_name
        )
        
        for to_adjust in adjust_list:
            change_logic += "_" + to_adjust

        remove_keys_from_dict(
            replacement_dict,
            adjust_list,
        )
        print(f"Replacement dict: {replacement_dict}")

    # hydra does not like keys starting with numbers
    replacement_dict = {f"_{k}": v for k, v in replacement_dict.items()}

    return replacement_dict, change_logic

def get_training_args(
        cfg: DictConfig,
        dataset: str,
        strategy_name: str,
        replacement_logic: str,
        replacement_dict: dict,
    ):
    args = {
        "model": "eq_nasnet",
        "+replacement_logic": replacement_logic,
    }
    dropout_rate = replacement_dict.get("-1_dropout_rate", None)
    expand_ratio = replacement_dict.get("-1_expand_ratio", None)
    if dropout_rate is not None:
        del replacement_dict["-1_dropout_rate"]
        args["model.dropout_rate"] = dropout_rate
    if expand_ratio is not None:
        del replacement_dict["-1_expand_ratio"]
        args["model.eq_expand_ratio"] = expand_ratio
    if len(replacement_dict) > 0:
        args["model.blocks_args_dict"] = convert_dict_to_hydra_string(replacement_dict)


    for key, value in cfg.additional_args.items():
        args[key] = value


    args["wandb.give_name"] = f"strategy_{strategy_name}{replacement_logic}"
    args["training"] = f"{dataset}-training"

    return args


def strategy_on_dataset(
        cfg: DictConfig,
        dataset: str,
        adjust_list: List[str], 
        strategy_name: str = "isic2019",
    ):    
    replacement_dict, adjusted = get_adjusted_dict(
        map_strategies_2_replacement_groups=cfg.map_strategies_2_replacement_groups,
        data_dir=cfg.data_dir,
        dataset_name=dataset,
        strategy_name=strategy_name,
        adjust_list=adjust_list,
    )
    
    args = get_training_args(
        cfg=cfg,
        dataset=dataset,
        strategy_name=strategy_name,
        replacement_logic=adjusted,
        replacement_dict=replacement_dict,
    )
        
    #print(f"Strategy dict: {strategy_dict}")
    print(f"Args: {args}")

    run_command(args, {}, test=False)


def iter_over_strategies(cfg: DictConfig):
    for adjust_list in cfg.adjust_lists:
        for dataset in cfg.datasets:
            strategy_on_dataset(
                cfg=cfg, 
                dataset=dataset, 
                adjust_list = adjust_list
            )


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    iter_over_strategies(cfg)


if __name__ == "__main__":
    main()