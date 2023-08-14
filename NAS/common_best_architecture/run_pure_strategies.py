import pandas as pd
import sys
import os
# issue with https://github.com/pytorch/pytorch/issues/37377
os.environ["MKL_THREADING_LAYER"]="GNU"
sys.path.append(f"{os.getcwd()}")
from NAS.util import encode_parameters
from experiment.run_files.run_command import run_command

choice_2_range_params = {
    "group": [1, 2, 4, 8, 16],
}

result_df = pd.read_csv('NAS/data/common_best_architecture/pure_strategies.csv', index_col=0)


global_args = ["model=eq_nasnet"]
for strategy_row in result_df.iterrows():
    strategy_name = "_".join(strategy_row[0].split('_')[1:])
    print(f"\nPure strategy: {strategy_name}")
    for dataset_row in result_df.iterrows():
        dataset_name = "_".join(dataset_row[0].split('_')[1:])
        print(f"Dataset: {dataset_name}")
        
        strategy_dict = strategy_row[1].to_dict()
        dataset_dict = dataset_row[1].to_dict()

        # we replace the (X_reflection, X_group) of strategy with the one of dataset
        for key in strategy_dict.keys():
            if '_reflection' in key or '_group' in key:
                strategy_dict[key] = dataset_dict[key]

        blocks_args = encode_parameters(strategy_dict, choice_2_range_params)

        # Create a new run
        args = [
            f"training={dataset_name}-training",
            f"wandb.tags=[eqnasnet,find_best,pure_strategy]",
            f"model.blocks_args={blocks_args}",
            f"model.dropout_rate={strategy_dict['-1_dropout_rate']}",
            f"model.eq_expand_ratio={strategy_dict['-1_expand_ratio']}",
            f"wandb.give_name=pure_strategy_{strategy_name}"
        ]

        # if strategy_name == "galaxy10" and dataset_name == "cifar10":
        #     run_command(args, global_args, test=False, path="experiment/")
        # elif strategy_name == "mnist_rot" and dataset_name == "galaxy10":
        #     args.append("training.dataset.batch_size=64")
        #     args.append("training.dataset.eval_batch_size=64")
        #     args.append("training.accumulate=4")
        #     run_command(args, global_args, test=False, path="experiment/")

        if dataset_name == "cifar10":
            args.append("training.dataset.rotation=True")
            args.append("training.dataset.name=cifar10_rot")
            run_command(args, global_args, test=False, path="experiment/")


