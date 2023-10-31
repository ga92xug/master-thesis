

def create_name_comparision_models(config):
    """
    Creates the name for the comparision between models from the wandb config.
    Example names: "eq_nasnet", "vit_pre", "vit", "efficientnet_pre", "efficientnet"
    """

    name = config["model"]["_target_"].split(".")[-1]
    pretrained = config["model"].get("pretrained", False)
    if pretrained:
        name += "_pre"
    return name