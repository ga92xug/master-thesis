import subprocess
import os
import sys
sys.path.append('experiment')
os.environ['HYDRA_FULL_ERROR'] = '1'
#print("Add path", pathlib.Path(sys.path[-1]).absolute())
#from experiment.speed_test import extract_info_instantiate_network

def run_command(args, global_args, test, path="experiment/"):
    global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1",
    ]

    if test == "instantiation":
        command = ["python", path + "model_instantiate.py", "-m"]
        command.extend(global_args_test)
        command.extend(args)
        subprocess.run(command)
    else:
        # we run through the main training loop
        command = ["python", path + "main.py", "-m"]
        if test:
            command.extend(global_args_test)
        else:
            command.extend(global_args)

        command.extend(args)
        subprocess.run(command)
    