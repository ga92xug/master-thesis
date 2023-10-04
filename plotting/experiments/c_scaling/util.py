from typing import List, Dict
import numpy as np

def replace_first_occurrence(input_str, search_str, replace_str):
    index = input_str.find(search_str)
    if index != -1:
        return input_str[:index] + replace_str + input_str[index + len(search_str):]
    return input_str

def baseline_and_scaling_exp_2_scaling_exp(
        exp_data: dict,
        label2sub_exp_name_dict: Dict[str, str],
    ):
    """Transforms the input data into the format required for create_subplot.

    Args:
    data: A dictionary containing accuracy and FLOPs data.

    Returns:
    transformed_data: A dictionary with transformed data.
    """
    transformed_data = {}
    # inverse the dictionary
    sub_exp_name2lable_dict = {v: k for k, v in label2sub_exp_name_dict.items()}

    for sub_exp_name, sub_exp_dict in exp_data.items():
        label_mask = sub_exp_name2lable_dict[sub_exp_name]

        print("sub_exp_dict.keys()", sub_exp_dict.keys())
        for group_name, group_dict in sub_exp_dict.items():
            if len(sub_exp_dict) == 1:
                # these sub experiments do not have different variations like w1.5,w2,w2.5
                label = label_mask
            else:
                print("group_name", group_name, type(group_name))
                if isinstance(group_name, tuple):
                    label = label_mask
                    for group in group_name:
                        label = replace_first_occurrence(label, "_", str(group))
                elif isinstance(group_name, str):
                    label_mask.replace("_", str(group_name))
                else:
                    raise ValueError(f"Unknown type of group_name: {type(group_name)}")

            if label in transformed_data:
                raise ValueError(f"Label {label} already exists in transformed_data.")
            transformed_data[label] = group_dict

    transformed_data = dict(sorted(transformed_data.items()))
    return transformed_data



def split_dict2lists(data, metric: str):
    """
    Splits the dictionary into lists for plotting.
    """

    #data = dict(sorted(data.items()))

    flops = []
    accuracy_values = []
    labels = []

    for label, values in data.items():
        print("values.keys()", values.keys())
        flops.append(values['flops'])
        accuracy_values.append(values[metric])
        labels.append(label)

    return flops, accuracy_values, labels


def example_data():
    flops = [np.array(7.06396968e+10), np.array(1.44022375e+11), np.array(2.42780298e+11), np.array(3.66913477e+11)]
    accuracy_values = [np.array([68.1474785 , 66.86897278, 66.55477285]), np.array([68.89136434, 68.97262534, 67.66067346]), np.array([70.24774353, 69.6485877 , 69.03163393]), np.array([69.07265186, 69.29402153, 66.99118813])]
    labels = ['w1', 'w1.5', 'w2', 'w2.5']

    return flops, accuracy_values, labels