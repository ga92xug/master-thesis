import subprocess

def run_command(args):
    command = ["python", "experiment/main.py", "-m"]
    command.extend(args)
    subprocess.run(command)