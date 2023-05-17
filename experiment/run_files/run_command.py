import subprocess

def run_command(args, global_args, test):
    global_args_test = global_args + [
        "wandb.mode=disabled",
        "training.steps_per_epoch=10",
        "training.epochs=1",
    ]
    if test == "instantiation":
        # only instantiate and test network
        command = ["python", "networks/network_instantiation.py", "-m"]
        command.extend(global_args_test)
        command.extend(args)
        output = subprocess.check_output(command, text=True)
        return output
    else:
        # we run through the main training loop
        command = ["python", "experiment/main.py", "-m"]
        if test:
            command.extend(global_args_test)
        else:
            command.extend(global_args)

        command.extend(args)
        subprocess.run(command)
    