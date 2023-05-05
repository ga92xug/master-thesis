import subprocess

def run_command(args):
    command = ["python", "experiment/main.py", "-m"]
    command.extend(args)
    subprocess.run(command)

def run_command(args, global_args):
    # "experiment/main.py"
    command = ["python", "experiment/main.py", "-m"]
    command.extend(args)
    command.extend(global_args)
    subprocess.run(command)

def run_command_cifar(args, global_args):
    command = ["python", "experiment/main.py", "-m"]
    command.extend(args)
    command.extend(global_args)
    subprocess.run(command)