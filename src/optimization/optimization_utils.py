from collections.abc import MutableMapping

def convert_dict_to_hydra_string(dictionary: dict):
    """
    Hydra demands a specific string format for dictionary overrides. This function recursively converts a dictionary to a string in the correct format.
    """
    key_value_comma = ",".join([f'{key}:{value if not isinstance(value, dict) else convert_dict_to_hydra_string(value)}' for key, value in dictionary.items()])
    return f'{{{key_value_comma}}}'

def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + str(separator) + str(key) if parent_key != "" else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)