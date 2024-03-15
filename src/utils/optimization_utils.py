def convert_dict_to_hydra_string(dictionary: dict):
    """
    Hydra demands a specific string format for dictionary overrides. This function recursively converts a dictionary to a string in the correct format.
    """
    key_value_comma = ",".join([f'{key}:{value if not isinstance(value, dict) else convert_dict_to_hydra_string(value)}' for key, value in dictionary.items()])
    return f'{{{key_value_comma}}}'