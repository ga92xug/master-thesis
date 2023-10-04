from typing import List
import numpy as np

def baseline_and_scaling_exp_2_scaling_exp(data):
    """Transforms the input data into the format required for create_subplot.

    Args:
    data: A dictionary containing accuracy and FLOPs data.

    Returns:
    transformed_data: A dictionary with transformed data.
    """
    transformed_data = {}

    keys = list(data.keys())
    assert len(keys) == 2, f"Expected two keys, got {len(keys)}"
    for key in keys:
        if "baseline" in key:
            baseline_key = key
        
    if baseline_key is None:
        raise ValueError("No baseline key found.")
    
    # other key is the scaling key
    scaling_key = [key for key in keys if key != baseline_key][0]

    # weight labels
    scaling_label = scaling_key[0]

    # Add the baseline data
    transformed_data[scaling_label + "1"] = data[baseline_key]["all"]

    for key, values in data[scaling_key].items():
        transformed_data[scaling_label + str(key)] = values

    return transformed_data



def transform_data_for_plot(data):
    """Transforms the input data into the format required for create_subplot.

    Args:
    data: A dictionary containing accuracy and FLOPs data.

    Returns:
    flops: A list of FLOPs values.
    accuracy_values: A list of lists containing accuracy values for each data point.
    labels: A list of labels for the points.
    """

    data = dict(sorted(data.items()))

    flops = []
    accuracy_values = []
    labels = []

    for label, values in data.items():
        flops.append(values['flops'])
        accuracy_values.append(values['valid.acc_weighted'])
        labels.append(label)

    return flops, accuracy_values, labels
