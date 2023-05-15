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
    if test == "instantiation":
        command = ["python", "networks/network_instantiation.py"]
    else:
        command = ["python", "experiment/main.py", "-m"]
    if test:
        global_args.extend(["wandb.mode=disabled"])
        if test == "quick":
            global_args.extend(["training.steps_per_epoch=10", "training.epochs=1"])
    command.extend(global_args)
    command.extend(args)
    subprocess.run(command)

def run_command_test(args, global_args):
    command = ["python", "networks/network_instantiation.py"]
    global_args.extend(args)
    command.extend([f"--overrides={global_args}"])
    subprocess.run(command)