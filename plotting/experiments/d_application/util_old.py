import wandb
from typing import Dict, List, Union
import matplotlib.colors as mcolors

def create_name_comparision_models(config):
    """
    Creates the name for the comparision between models from the wandb config.
    Example names: "eq_nasnet", "vit_pre", "vit", "efficientnet_pre", "efficientnet"
    """

    name = config["model"]["_target_"].split(".")[-1]
    pretrained = config["model"].get("pretrained", False)
    if pre_trained
        name += " pre-trained"

    if "EquivariantNASNet" in name:
        name = "Eq-NASNet"
    return name

def sorting_key(name: str):
    sort_key = None
    if "EfficientNet" in name:
        sort_key = 10
    elif "ViT" in name:
        sort_key = 20
    elif "Eq-NASNet" in name:
        sort_key = 30

    if "pre-trained" in name:
        sort_key += 1

    if sort_key is None:
        sort_key = 100

    #print(name, "sort_key", sort_key)
    return sort_key



def get_wandb_runs_from_filters(
        filters: Dict[str, str],
        project: str = "low_data",
        entity: str = "ga92xug",
    ):
    api = wandb.Api()
    filters["state"] = "finished"
    runs = api.runs(path=f"{entity}/{project}", filters=filters)

    print("Number of runs:", len(runs))
    return runs

def name2color(name: str):
    name2color_dict = {
        "Eq-NASNet": mcolors.CSS4_COLORS["red"],
        "ViT": mcolors.CSS4_COLORS["blue"],
        "ViT pre-trained": mcolors.CSS4_COLORS["navy"],
        "EfficientNet": mcolors.CSS4_COLORS["limegreen"],
        "EfficientNet pre-trained": mcolors.CSS4_COLORS["green"],
    }
    return name2color_dict[name]