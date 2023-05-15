import subprocess

def run_command(args):
    command = ["python", "experiment/main.py", "-m"]
    command.extend(args)
    subprocess.run(command)

def run_command(args, global_args):
    command = ["python", "experiment/main.py", "-m"]
    command.extend(args)
    command.extend(global_args)
    subprocess.run(command)

def run_command(args, global_args, test):
    global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1",
    ]

    if test == "instantiation":
        command = ["python", "networks/network_instantiation.py", "-m"]

    else:
        command = ["python", "experiment/main.py", "-m"]

    if test:
        command.extend(global_args_test)
    else:
        command.extend(global_args)

    command.extend(args)
    subprocess.run(command)

def run_command_test(args, global_args):
    command = ["python", "networks/network_instantiation.py"]
    global_args.extend(args)
    command.extend([f"--overrides={global_args}"])
    subprocess.run(command)