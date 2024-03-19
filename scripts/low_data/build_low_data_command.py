import submitit
import os

def build_slurm_command_low_data(
        experiment_location, 
        args,
        execute=False,
        local=False,
        submitit_logs="/home/atuin/b180dc/b180dc27/logs/submitit_logs",
    ):        
    if not local:
        executor = submitit.AutoExecutor(folder=submitit_logs)

        executor.update_parameters(
            nodes=1,
            tasks_per_node=1,
            slurm_max_num_timeout=2,
            slurm_time="16:00:00",  # each job takes max
            slurm_gres="gpu:a40:1",
            slurm_partition="a40",
            cpus_per_task=1,
        )
    else:
        # https://github.com/facebookincubator/submitit/issues/1718
        # would run all jobs at once -> not wanted
        pass

    networks = [
        "efficientnet_pre",
        "efficientnet",
        "vit_pre",
        "vit",
        "eq_nasnet_pre",
        "eq_nasnet",
    ]
    networks = [experiment_location + experiment for experiment in networks]
    seeds = "seed=0,1,2,3,4"

    for network in networks:
        for rf, epochs, patience, val_check in args:
            job_name = f"{network.split('/')[-2]}_{network.split('/')[-1]}_rf{rf}"

            command = network
            command += f" train.dataset.reduction_factor={rf}"
            command += f" train.trainer.max_epochs={epochs}"
            command += f" train.callbacks.early_stopping.patience={patience}" 
            command += f" train.trainer.check_val_every_n_epoch={val_check}"
            command += f" {seeds}"

            if not execute:
                print(command)
                continue

            if local:
                job_runner(command)
            else:
                executor.update_parameters(slurm_job_name=job_name)
                executor.submit(job_runner, command)


def job_runner(command, multirun=True):
    activate_command = "source activate scaling"
    os.system(activate_command)
    print("Virtual environment activated")

    # Your SLURM job commands translated into Python
    py_script = "python src/main.py"
    if multirun:
        py_script += " -m"

    # Construct and execute the command
    full_command = f"{py_script} {command}"
    os.system(full_command)

