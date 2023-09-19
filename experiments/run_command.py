import subprocess
import os
import sys

from experiments.util import convert_dict_to_hydra_string
sys.path.append('experiment')
os.environ['HYDRA_FULL_ERROR'] = '1'
#print("Add path", pathlib.Path(sys.path[-1]).absolute())
#from experiment.speed_test import extract_info_instantiate_network

def run_command(
        args, 
        global_args, 
        test, 
        path:str = "training/"
    ):
    assert (isinstance(args, list) and isinstance(global_args, list)) \
        or (isinstance(args, dict) and isinstance(global_args, dict)), \
        "args and global_args must be either both lists or both dicts"
    
    if isinstance(args, list):
        # convert to dict
        args = {k.split("=")[0]: k.split("=")[1] for k in args}
        global_args = {k.split("=")[0]: k.split("=")[1] for k in global_args}

    # check if args and global_args have the same keys
    duplicate_keys = set(args.keys()).intersection(set(global_args.keys()))
    if duplicate_keys:
        print("Warning: duplicate keys in args and global_args:")
        print(duplicate_keys)
    
    # combine args and global_args, args have priority
    command_args_dict = {**global_args, **args}

    if test:
        command_args_dict["other.debug"] = True 

    # convert to list
    command_args = [f"{k}={v if not isinstance(v, dict) else convert_dict_to_hydra_string(v)}" for k, v in command_args_dict.items()]


    if test == "instantiation":
        # we only instantiate the model
        command = ["python", path + "model_instantiate.py", "-m"]
    else:
        # we run through the main training loop
        command = ["python", path + "main.py", "-m"]
    
    command.extend(command_args)
    subprocess.run(command, check=True)

    