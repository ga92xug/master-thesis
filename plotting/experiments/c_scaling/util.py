from typing import List, Dict
import numpy as np
import re

def replace_first_occurrence(input_str, search_str, replace_str):
    index = input_str.find(search_str)
    if index != -1:
        return input_str[:index] + replace_str + input_str[index + len(search_str):]
    return input_str

def extract_number(key):
    # Use regular expression to extract the numeric part and decimal part
    match = re.search(r'(\d+(\.\d+)?)', key)
    if match:
        return float(match.group())
    return 0.0  # Return 0.0 if there are no numbers in the key

def baseline_and_scaling_exp_2_scaling_exp(
        wandb_data: dict,
        exp_dict: Dict[str, str],
    ):
    
    """
    Transforms the input data into the format required for create_subplot.

    Args:
        exp_dict: dict
            paths: dict
                label_mask: path_name
            colors: list
            linestyles: list


    Returns:
    transformed_data: A dictionary with transformed data.
    """
    transformed_data = {}
    # inverse the dictionary
    path2lable_mas_dict = {v: k for k, v in exp_dict["paths"].items()}

    #print("exp_data", exp_data)

    for path_name, path_wandb_data in wandb_data.items():
        label_mask = path2lable_mas_dict[path_name]

        transformed_data[path_name] = {}
        for group_name, group_dict in path_wandb_data.items():
            if len(path_wandb_data) == 1:
                # these paths do not have different variations like w1.5,w2,w2.5
                label = label_mask
            else:
                if isinstance(group_name, tuple):
                    label = label_mask
                    for group in group_name:
                        label = replace_first_occurrence(label, "_", str(group))
                elif isinstance(group_name, (str, int, float)):
                    label = label_mask.replace("_", str(group_name))
                else:
                    raise ValueError(f"Unknown type of group_name: {type(group_name)}")

            if label in transformed_data[path_name]:
                raise ValueError(f"Label {label} already exists in transformed_data.")
            transformed_data[path_name][label] = group_dict

    #sorted_dict = dict(sorted(transformed_data.items(), key=lambda item: extract_number(item[0])))

    sorted_dict = {}
    for k, v in transformed_data.items():
        dict(sorted(v.items(), key=lambda item: extract_number(item[0])))
        sorted_dict[k] = dict(sorted(v.items(), key=lambda item: extract_number(item[0])))

    return sorted_dict



def split_dict2lists(data, metric: str):
    """
    Splits the dictionary into lists for plotting.
    """

    #data = dict(sorted(data.items()))

    flop_list = []
    accuracy_list = []
    label_list = []

    for path, path_dict in data.items():
        flop = []
        accuracy = []
        labels = []
        for label, values in path_dict.items():
            flop.append(values['flops'])
            accuracy.append(values[metric])
            labels.append(label)

        flop_list.append(flop)
        accuracy_list.append(accuracy)
        label_list.append(labels)

    return flop_list, accuracy_list, label_list


def example_data():
    flops = [np.array(7.06396968e+10), np.array(1.44022375e+11), np.array(2.42780298e+11), np.array(3.66913477e+11)]
    accuracy_values = [np.array([68.1474785 , 66.86897278, 66.55477285]), np.array([68.89136434, 68.97262534, 67.66067346]), np.array([70.24774353, 69.6485877 , 69.03163393]), np.array([69.07265186, 69.29402153, 66.99118813])]
    labels = ['w1', 'w1.5', 'w2', 'w2.5']

    return flops, accuracy_values, labels