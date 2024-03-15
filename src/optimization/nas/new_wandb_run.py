# wandb_run.py
import argparse
import wandb

def create_wandb_run(entity, project, mode, trial_index):
    wandb_run = wandb.init(
        entity=entity,
        project=project,
        mode=mode,
        name=str(trial_index),
    )
    wandb.finish()
    print("Wandb run id:", wandb_run.id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="wandb entity")
    parser.add_argument("--project", required=True, help="wandb project")
    parser.add_argument("--mode", required=True, help="wandb mode")
    parser.add_argument("--trial_index", type=int, required=True, help="wandb trial index")

    args = parser.parse_args()
    create_wandb_run(args.entity, args.project, args.mode, args.trial_index)
