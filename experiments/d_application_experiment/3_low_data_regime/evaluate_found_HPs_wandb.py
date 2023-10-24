from typing import Dict
import wandb
import yaml
import os


def get_runs(dataset2metrics: Dict):
    api = wandb.Api()

    filters = {
        "state": "finished",
        "config.wandb.tags": {"$regex": ".*low_data.*"},
    }

    runs = api.runs(path=f"ga92xug/SL-ray", filters=filters)

    results = {}

    for run in runs:
        #print("\nCurrent name:", run.config["wandb"]["tags"])

        low_data = run.config["training"]["dataset"]["reduction_factor"]
        dataset = run.config["training"]["dataset"]["name"]
        model = run.config["model"]["_target_"].split(".")[-1]
        pretrained = run.config["model"].get("pretrained", False)
        model_name = f"{model}_pre" if pretrained else model
        metric_name = dataset2metrics[dataset]
        metric = extract_metric(run, metric_name)
        
        if dataset not in results:
            results[dataset] = {}

        if low_data not in results[dataset]:
            results[dataset][low_data] = {}

        if model_name not in results[dataset][low_data]:
            # no run with this model and low_data yet
            
            results[dataset][low_data][model_name] = {
                "config": run.config, 
                "id": run.id,
                metric_name: metric,
            }
        else: 
            # run with this model and low_data already exists
            if metric > results[dataset][low_data][model_name][metric_name]:
                results[dataset][low_data][model_name] = {
                    "config": run.config, 
                    "id": run.id,
                    metric_name: metric,
                }

    return results
        
def get_nested(dict_obj, key_str):
    keys = key_str.split('.')
    for key in keys:
        dict_obj = dict_obj[key]
    return dict_obj

def extract_metric(run, metric_name, mode="last"):
    if mode == "last":
        return get_nested(run.summary, metric_name) 
    elif mode == "max":
        history = run.scan_history(keys=[metric_name])
        metric_hist = [row[metric_name] for row in history]
        return max(metric_hist)
    else:
        raise ValueError(f"Unknown mode {mode}.")


# path to current file
def get_dataset2metrics():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    dataset2metrics = {}

    for _file in os.listdir(file_dir + '/dataset'):
        dataset_name = _file.split('.')[0]
        if dataset_name == 'DeepDRiD':
            dataset_name = 'DeepDRiD_quality'
        file_name = file_dir + '/dataset/' + _file
        with open(file_name, 'r') as file:
            yaml_dict = yaml.safe_load(file)

        dataset2metrics[dataset_name] = yaml_dict["BO_optimizer"]["metric"]

    print("dataset2metrics", dataset2metrics)
    return dataset2metrics


def main():
    dataset2metrics = get_dataset2metrics()
    results = get_runs(dataset2metrics)
    print(results)


if __name__ == "__main__":
    main()
