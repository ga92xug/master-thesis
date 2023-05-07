import sys

from matplotlib import pyplot as plt
sys.path.append('../scaling-laws-ecnn') # add parent directory

from networks.network_instantiation import run


def binary_search_over_model_scaling(
        search_param, scale_param, scaling_factor, initial_range, constraint_func,
        overrides, global_overrides, max_iterations=50, tolerance=0.01):
    """
    Perform a binary search to find the best scaling parameter for a given model configuration.

    Args:
        search_param (dict): A dictionary containing the parameter to search for (key) and its initial value (value).
        scale_param (dict): A dictionary containing the parameter to scale for (key) and its initial value (value).
        scaling_factor (float): The desired scaling factor for the scale_param.
        initial_range (tuple): A tuple containing the initial lower and upper bounds for the search_param.
        constraint_func (function): A function to apply model-specific constraints to the search_param.
        overrides (dict): A dictionary containing model configuration parameters that are not part of the search.
        global_overrides (dict): A dictionary containing global configuration parameters for the model.
        max_iterations (int, optional): The maximum number of iterations for the binary search. Default is 100.
        tolerance (float, optional): The acceptable error threshold for convergence. Default is 0.01.

    Returns:
        mid: The best scaling parameter found for the search_param.

    Raises:
        ValueError: If the search_param or scale_param is not valid or supported.
    """
    scale_indices = {
        "param_count": 0,
        "model_building_time": 1,
        "train_time": 2
    }
    
    search_key = list(search_param.keys())[0]
    scale_key = list(scale_param.keys())[0]
    scale_index = scale_indices[scale_key]

    initial_value = search_param[search_key]
    local_overrides = {**overrides, **search_param}
    total_overrides = global_overrides.copy()
    total_overrides.update(local_overrides)

    base_stats = run(overrides=total_overrides, verbose=0)

    lower_bound, upper_bound = initial_range

    is_integer = isinstance(initial_value, int)

    iteration = 0
    while iteration < max_iterations:
        mid = (lower_bound + upper_bound) / 2
        if is_integer:
            mid = int(mid)

        mid = constraint_func(mid)
        local_overrides[search_key] = mid
        total_overrides.update(local_overrides)

        try:
            stats = run(overrides=total_overrides, verbose=0)
        except:
            upper_bound = mid * 0.9
            continue

        if abs(stats[scale_index] - base_stats[scale_index] * scaling_factor) <= tolerance:
            break
        elif stats[scale_index] > base_stats[scale_index] * scaling_factor:
            upper_bound = mid * 0.9
        else:
            lower_bound = mid * 1.1

        iteration += 1

    print(f"Best {search_key} found: {mid}")
    return mid


def plot_model_data(model_data, vs_param):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    fig.suptitle(f'Model Performance vs {vs_param}', fontsize=14, fontweight='bold')
    
    for model_name, data in model_data.items():
        vs = data[:, 0]
        param_count = data[:, 1]
        model_building_time = data[:, 2]
        train_time = data[:, 3]
        
        ax[0].plot(vs, param_count, label=model_name)
        ax[1].plot(vs, model_building_time, label=model_name)
        ax[2].plot(vs, train_time, label=model_name)
    
    ax[0].set_xlabel(f'{vs_param}')
    ax[0].set_ylabel('Parameter Count')
    ax[0].set_title('Parameter Count vs {vs_param}')
    ax[0].legend()

    ax[1].set_xlabel(f'{vs_param}')
    ax[1].set_ylabel('Model Building Time')
    ax[1].set_title(f'Model Building Time vs {vs_param}')
    ax[1].legend()

    ax[2].set_xlabel(f'{vs_param}')
    ax[2].set_ylabel('Train Time')
    ax[2].set_title(f'Train Time vs {vs_param}')
    ax[2].legend()

    plt.tight_layout()
    plt.show()


