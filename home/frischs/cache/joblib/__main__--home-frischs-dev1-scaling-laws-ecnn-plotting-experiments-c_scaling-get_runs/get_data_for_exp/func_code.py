# first line: 127
@memory.cache
def get_data_for_exp(
        paths_dict: dict,
        wandb_entity: str,
        wandb_projects: str,
        metric: str,
    ):
    """
    Get all the data for 1 experiment. 1 experiment is a set of paths experiments with labels.
    Each path has a set of filters that are used to get the right data from wandb.
    Optionally, the data can be grouped by a certain key.
    A group would for example be the width coefficient of the model.
    """

    data = {}
    for path_name, value_dict in paths_dict.items():
        filters = value_dict["filters"]
        group_by = value_dict["group_by"]

        # add config.training.epochs: 120 to filters
        filters["config.training.epochs"] = 120
        
        results_for_filter = get_wandbdata_with_filters(
            wandb_entity=wandb_entity,
            wandb_projects=wandb_projects,
            filters=filters,
            metric=metric,
            group_by=group_by,
        )
        data[path_name] = results_for_filter

    return data
