from typing import Dict

from networks.eq_nasnet.util import round_repeats


def get_scaling_name(config: Dict) -> str:
    """
    Returns the scaling name for the given config.
    """
    model_config = config["model"]

    # blocks
    num_blocks = get_num_blocks_eq_nasnet(model_config)

    # depth
    num_layer_blocks = model_config.get("num_layer_blocks", None)
    if num_layer_blocks is None:
        num_layer_blocks = get_num_layer_blocks(model_config, num_blocks)

        print(f"WARNING: num_layer_blocks not found in model config. Update config {num_layer_blocks}.")
        model_config["num_layer_blocks"] = num_layer_blocks

    # width
    width_coefficient = float_to_int_if_possible(model_config["width_coefficient"])
    # resolution
    resolution = config["training"]["dataset"]["resolution"]
    scaling_name = f"b{num_blocks}_d{num_layer_blocks}_w{width_coefficient}_r{resolution}"
    return scaling_name

def get_num_blocks_eq_nasnet(model_config: Dict) -> int:
    """
    Returns the number of blocks of the EquivariantNASNet model.
    """
    # every model should have a num_blocks parameter
    num_blocks = model_config.get("num_blocks", None)

    # legacy
    if num_blocks is None:
        print("WARNING: num_blocks not found in model config. Using legacy convention.")
        try:
            num_blocks = model_config["increase_blocks"]["2"]["num_new_blocks"] + 3
        except KeyError:
            num_blocks = 3

    return num_blocks

def get_num_layer_blocks(model_config: Dict, num_blocks: int) -> str:
    """
    Returns the depth name for the given config.
    """
    blocks_args_dict = model_config["blocks_args_dict"]
    depth_coefficient = model_config["depth_coefficient"]
    not_increase_1_layer = model_config.get("not_increase_1_layer", False)

    num_layers_blocks = ""
    for i in range(1, num_blocks+1):
        num_layers_block = int(blocks_args_dict[f"_{i}"]["num_layers"])

        assert not (num_layers_block >= 3 and depth_coefficient != 1.0), \
            f"num_layers_block: {num_layers_block}, depth_coefficient: {depth_coefficient}"
        
        num_layers_block = round_repeats(num_layers_block, depth_coefficient, not_increase_1_layer)

        num_layers_blocks += "-" + str(num_layers_block)

    return num_layers_blocks

def float_to_int_if_possible(x):
    if x == int(x):
        return int(x)
    return x