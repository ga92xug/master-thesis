import wandb
from typing import Dict, List, Union

def create_name_comparision_models(config):
    """
    Creates the name for the comparision between models from the wandb config.
    Example names: "eq_nasnet", "vit_pre", "vit", "efficientnet_pre", "efficientnet"
    """

    name = config["model"]["_target_"].split(".")[-1]
    pretrained = config["model"].get("pretrained", False)
    if pretrained:
        name += " pre-trained"
    return name


def get_wandb_runs_from_filters(
        filters: Dict[str, str],
        project: str = "SL-Application",
        entity: str = "ga92xug",
    ):
    api = wandb.Api()
    filters["state"] = "finished"
    runs = api.runs(path=f"{entity}/{project}", filters=filters)

    print("Number of runs:", len(runs))
    return runs