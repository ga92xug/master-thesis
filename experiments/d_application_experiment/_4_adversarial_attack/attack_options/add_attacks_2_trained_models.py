import wandb
import torch
import torch.nn as nn
import sys
import os


sys.path.append(f"{os.getcwd()}")
from experiments.d_application_experiment._4_adversarial_attack.adversarial_attack import adversarial_attack
from training.model_instantiate import hydra_compose 

models2runids = {
    "DeepDRiD": {
        "ViT_pre": "dc4md2zl",
        "ViT": "r5ck9f6m",
        "eq_nasnet": "myw54oac",
        "EfficientNet_pre": "jwuyjulk",
        "EfficientNet": "xe6hgfkd",
    },
    "blood": {
        "ViT_pre": "6sf7aw46",
        "ViT": "xfmi9gmi",
        "eq_nasnet": "ra78gixb",
        "EfficientNet_pre": "1pud3wdh",
        "EfficientNet": "25zaw08u",
    },
}

def load_model(model: nn.Module, run_id: str):
    path = f"/home/frischs/outputs/{run_id}/model.pth"
    model.load_state_dict(torch.load(path))


def add_attacks_to_trained_model(
        device: torch.device,
        model_name: str,
        dataset_name: str,
        run_id: str,
    ):
    """
    This function can be used to debug the auto_attack_eval function.
    """

    # wandb run
    run = wandb.init(
        entity="ga92xug",
        project="SL-Application",
        id=run_id, 
        resume="must",
    )
    run.finish()
    return
    
    model, dataloaders, cfg = hydra_compose(
        overrides=[f"training={dataset_name}-training", f"model={model_name}"])

    # load model
    load_model(model, run_id)

    results = adversarial_attack(
        mode="Foolbox",
        model=model,
        dataloader=dataloaders["test"],
        cfg=cfg,
        device=device,
    )
    results = {"adversarial_attack": results}

    #quit()
    run.log(results)
    run.finish()
    #run.summary.update(results)


def model_name2model_cfg_name(model_name: str):
    """
    Brings the model name into the same format as the hydra config names.
    """

    # remove _pre
    model_cfg_name = model_name.replace("_pre", "")

    # lower case
    model_cfg_name = model_cfg_name.lower()

    return model_cfg_name
    

def iterate_over_datasets_models():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    for dataset_name, models in models2runids.items():
        for model_name, run_id in models.items():
            if dataset_name != "DeepDRiD" and "ViT" not in model_name:
                continue

            print(f"dataset: {dataset_name}, model: {model_name}, run_id: {run_id}")
            model_cfg_name = model_name2model_cfg_name(model_name)

            add_attacks_to_trained_model(
                device=device,
                model_name=model_cfg_name,
                dataset_name=dataset_name,
                run_id=run_id,
            ) 

    

if __name__ == "__main__":
    iterate_over_datasets_models()

