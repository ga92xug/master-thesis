import re
from typing import List
import pandas as pd

from ax.service.ax_client import AxClient
from ax.service.utils.report_utils import exp_to_df

import sys
import os
sys.path.append(f"{os.getcwd()}")
from experiments.b_NAS.util import *

def optimal_experiment_trials(
        client: AxClient, 
        percentage: float = 0.02, 
        top_n: int = 10,
    ):
    """
    Pareto optimal experiment result and top n highest valid acc 

    Parameters:
        client (AxClient): The experiment client used to retrieve data.
        percentage (float): The percentage threshold for filtering.
        top_n (int): The number of top rows to append to the final DataFrame.

    Returns:
        pd.DataFrame
    """
    optimize_for = "valid_acc" if "valid_acc" in client.experiment.metrics else "valid_acc_weighted"

    # Convert the experiment to a DataFrame
    experiment_df = exp_to_df(client.experiment)
    experiment_df = experiment_df.drop_duplicates(subset=['arm_name'], keep=False)

    # Calculate the maximum valid_acc value and the threshold
    max_valid_acc = experiment_df[optimize_for].max()
    threshold = percentage * max_valid_acc

    # Get pareto optimal parameters with model predictions set to False
    pareto_optimal_parameter = client.get_pareto_optimal_parameters(use_model_predictions=False)
    pareto_optimal_trial_index = pareto_optimal_parameter.keys()
    indices_to_filter = [index for index in experiment_df["trial_index"] if index in pareto_optimal_trial_index]
    pareto_optimal_df = experiment_df[experiment_df["trial_index"].isin(indices_to_filter)]

    # Get the top n highest validation accuracy rows
    top_n_high_acc_rows = experiment_df.nlargest(top_n, optimize_for)

    # Append the top 10 rows to the final filtered DataFrame
    final_df = pd.concat([pareto_optimal_df, top_n_high_acc_rows])

    # new DataFrame containing only rows with valid_acc within the threshold and delete duplicates
    final_filtered_df = final_df[abs(final_df[optimize_for] - max_valid_acc) <= threshold]
    final_filtered_df = final_filtered_df.drop_duplicates(subset=['trial_index'], keep='first')

    # Sort the final DataFrame by valid_acc
    final_filtered_df = final_filtered_df.sort_values(by=[optimize_for], ascending=False)

    return final_filtered_df

def get_weights_for_weighted_average(
        df: pd.DataFrame, 
        valid_acc_impact: float = 0.5, 
        epsilon: float = 0.01,
    ):
    optimize_for = "valid_acc" if "valid_acc" in df.columns else "valid_acc_weighted"

    valid_acc_obs = df[optimize_for]
    gflops_obs = df['gflops']

    # normalize valid_acc_weights and gflops_weights to [epsilon, 1]
    valid_acc_weights = epsilon + (1 - epsilon) * (valid_acc_obs - valid_acc_obs.min()) / (valid_acc_obs.max() - valid_acc_obs.min())
    gflops_weights = epsilon + (1 - epsilon) * (gflops_obs - gflops_obs.min()) / (gflops_obs.max() - gflops_obs.min())
    #print("valid_acc_weights", valid_acc_weights)
    #print("gflops_weights", gflops_weights)
    
    weights = valid_acc_impact * valid_acc_weights + (1 - valid_acc_impact) * gflops_weights
    return weights / weights.sum()

def summarize_optimal_architectures(
        df: pd.DataFrame, 
        valid_acc_weight: float = 0.8, 
        exclude_columns: List[str] = ['trial_index', 'arm_name', 'trial_status', 
                     'generation_method', 'model_building_time', 'is_feasible'],
    ):
    #optimize_for = "valid_acc" if "valid_acc" in df.columns else "valid_acc_weighted"

    weights = get_weights_for_weighted_average(df, valid_acc_weight)

    summary = {}
    for col in df.columns:
        if col not in exclude_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary[col] = (df[col] * weights).sum()
            else:
                counts = df.groupby(col).apply(lambda x: (weights.loc[x.index]).sum())
                summary_string = ""
                for value, percentage in (counts / counts.sum() * 100).items():
                    summary_string += f"{value}={int(percentage)}%; "
                summary[col] = summary_string

    summary_df = pd.DataFrame(summary, index=[f'weighted_average'])
    return summary_df

def get_categories(df: pd.DataFrame, round_to_float_01_mask: List[str] = ["-1_dropout_rate"]):
    categorical_mask = ["conv_op", "skip_op"]
    round_to_float_01_mask = round_to_float_01_mask
    round_to_float_025_mask = ["se_ratio", "out_channels"]
    donot_round_params = ["gflops", "valid_acc(_weighted)"]
    params = {
        "no_round": donot_round_params,
        "round_to_int": [],
        "round_to_float_025": [],
        "categorical": [],
        "round_to_float_01": []
        }

    for param in df.columns.tolist():
        if sum([1 for k, v in params.items() if param in v]):
            print(f"Skipping {param} because it is already assigned")   
            continue
        
        layer = param.split("_")[0]
        variable = param.split("_")[1:]
        variable = "_".join(variable)

        if variable in categorical_mask:
            params["categorical"].append(param)
        elif variable in round_to_float_01_mask:
            params["round_to_float_01"].append(param)
        elif variable in round_to_float_025_mask:
            params["round_to_float_025"].append(param)
        else:
            params["round_to_int"].append(param)

    return params


def apply_transformations(df: pd.DataFrame, **kwargs):
    params = get_categories(df, **kwargs)
    print("params", params)

    def round_to_nearest_multiple(x, base):
        return round(x / base) * base

    def extract_max_category(probability_string):
        #print("probability_string", probability_string)
        categories = re.findall(r'(\w+)=(\d+)%', probability_string)
        if not categories:
            return probability_string
        max_category = max(categories, key=lambda x: int(x[1]))
        return max_category[0]

    for category, columns in params.items():
        if category == 'no_round':
            continue
        elif category == 'round_to_int':
            for column in columns:
                df[column] = df[column].apply(lambda x: round(x))
        elif "round_to_float" in category:
            float_to = category.split("_")[-1]
            if float_to == "025":
                for column in columns:
                    df[column] = df[column].apply(lambda x: round_to_nearest_multiple(x, 0.25))
            elif float_to == "01":
                for column in columns:
                    df[column] = df[column].apply(lambda x: round_to_nearest_multiple(x, 0.1))
            else:
                raise ValueError(f"float_to {float_to} not supported")
        elif category == 'categorical':
            for column in columns:
                df[column] = df[column].apply(lambda x: extract_max_category(x))

    df.index = ["pure_" + str(index) for index in df.index]
    return df


def add_to_csv(to_add_df: pd.DataFrame, save_folder: str, filename: str):
    """
    Add a DataFrame to a existing csv file.
    """
    location = save_folder + filename + '.csv'
    old_df = pd.read_csv(location, index_col=0)
    assert old_df.columns.to_list() == to_add_df.columns.to_list()
    assert set(old_df.index).intersection(set(to_add_df.index)) == set()
    to_add_df = pd.concat([to_add_df, old_df], axis=0)
    to_add_df.to_csv(location, index=True, header=True)
    return to_add_df