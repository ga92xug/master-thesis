import os.path
import sqlite3
import pandas as pd
import numpy as np
import io
import datetime
import hydra
import signal

from typing import List

import torch

#from models import *
from networks import *
from networks.util import get_param_count

################################################################################
# building the model
################################################################################



def build_model(cfg, n_inputs, n_outputs, device, log, is_nas=False, trial_data=None):
    if not is_nas:
        # normal model building

        # build the model
        model = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
            image_size=cfg.dataset.resolution,
        ).to(device)
        if device != torch.device("cpu"):
            model = torch.nn.DataParallel(model)
        if cfg.training.compile:
            model = torch.compile(model)
        print("Stage 2: model built")

    else:
        # NAS model building

        

        try:
            # Your command that builds the model
            # Place the command here that occasionally takes a long time

            # build the model
            model = hydra.utils.instantiate(
                cfg.model,
                input_channels=n_inputs,
                num_classes=n_outputs,
                image_size=cfg.dataset.resolution,
            ).to(device)
            if device != torch.device("cpu"):
                model = torch.nn.DataParallel(model)
            if cfg.training.compile:
                model = torch.compile(model)

            # Cancel the alarm since the command finished before the timeout
            signal.alarm(0)
        except TimeoutError:
            # Handle the timeout error
            
            print("Command execution timed out")

        except torch.cuda.CudaError:
            print("Cuda out of memory")



def allowed_usage_time(
        respect_start_time: bool,
        start_time: datetime.time = datetime.time(hour=8, minute=30),
        end_time: datetime.time = datetime.time(hour=20),
):
    """
    GPU sharing. Check if the current time is within the allowed usage time.
    """
    if not respect_start_time:
        return

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).time()


def print_results(acc, loss, duration, mode, epoch, verbose):
    if verbose:
        print('-'*100)
        print(f'{mode} Epoch: {epoch} lasted {duration:.3f} seconds')
        print(f'Accuracy: {acc:.3f}; Loss: {loss:.3f}\n')


########################################################################################################################
# Utilites to build paths and names in a standard way
########################################################################################################################

def exp_name(cfg):
    values = []
    # model name
    if "EquivariantWideResNet" in cfg.model._target_: 
        values.append("eq_wrn")
    elif "EquivariantResNet9" in cfg.model._target_:
        values.append("res9")
    elif "EquivariantMobileNetV2" in cfg.model._target_:
        values.append("eq_mobv2")
    elif "WideResNet" in cfg.model._target_: 
        values.append("wrn")
    else:
        ValueError("Unknown model")

    # depth
    if "EquivariantWideResNet" in cfg.model._target_ or "WideResNet" in cfg.model._target_:
        values.append(f"{cfg.model.depth}")
    elif "EquivariantMobileNetV2" in cfg.model._target_ :
        values.append(f"{cfg.model.depth_multiplier}")
    
    # width
    if "EquivariantWideResNet" in cfg.model._target_ or "WideResNet" in cfg.model._target_:
        values.append(f"{cfg.model.widen_factor}")
    elif "EquivariantMobileNetV2" in cfg.model._target_ :
        values.append(f"{cfg.model.width_multiplier}")
    
    # kernel size
    if "EquivariantWideResNet" in cfg.model._target_ or "WideResNet" in cfg.model._target_:
        if len(cfg.model.kernel_layout) == 3:
            values.append(f"B({cfg.model.kernel_layout[0]},{cfg.model.kernel_layout[1]},{cfg.model.kernel_layout[2]})")
        elif len(cfg.model.kernel_layout) == 2:
            values.append(f"B({cfg.model.kernel_layout[0]},{cfg.model.kernel_layout[1]})")
        elif len(cfg.model.kernel_layout) == 1:
            values.append(f"B({cfg.model.kernel_layout[0]})")
        elif len(cfg.model.kernel_layout) == 4:
            values.append(f"B({cfg.model.kernel_layout[0]},{cfg.model.kernel_layout[1]},{cfg.model.kernel_layout[2]},{cfg.model.kernel_layout[3]})")

    if "Equivariant" in cfg.model._target_ and "nas" not in cfg.model._target_:
        if cfg.model.group == "cyclic":
            group = "C"
        elif cfg.model.group == "dihedral":
            group = "D"
        values.append(f"{group}{cfg.model.rotation}")
    # assert len(values) > 1, f"Experiment name should be at least a model and dataset, provided {values}"

    return "_".join(values)


def out_path(cfg):
    path = cfg.other.output_path
    return path


def plot_path(config):
    return os.path.join(out_path(config), exp_name(config) + ".svg")


def backup_path(config):
    backup_folder = os.path.join(out_path(config), exp_name(config))
    return os.path.join(backup_folder, f"_{config.other.seed}.model")

########################################################################################################################
# utilites to build dataloaders
########################################################################################################################

from experiment.datasets.mnist_rot import data_loader_mnist_rot
from experiment.datasets.mnist_fliprot import data_loader_mnist_fliprot
from experiment.datasets.mnist12k import data_loader_mnist12k
from experiment.datasets.cifar10 import data_loader_cifar10
from experiment.datasets.cifar100 import data_loader_cifar100
# from experiment.datasets.STL10 import data_loader_stl10
# from experiment.datasets.STL10 import data_loader_stl10frac
from experiment.datasets.imagenette import data_loader_imagenette
from experiment.datasets.Galaxy10_DECals import data_loader_Galaxy10_DECals



def build_dataloaders(cfg):
    dataset = cfg.training.dataset.name
    batch_size = cfg.training.dataset.batch_size
    num_workers = cfg.training.dataset.workers
    drop_last_train = cfg.training.dataset.drop_last_train or False
    augment = cfg.training.dataset.augment or False
    validation = cfg.training.earlystop or True 
    reshuffle = cfg.training.dataset.reshuffle or False
    eval_batch_size = cfg.training.dataset.eval_batch_size or None
    interpolation = cfg.training.dataset.interpolation or 2
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
        
    if dataset == "mnist_rot":
        
        if validation:
            if reshuffle:
                seed = np.random.randint(0, 100000)
            else:
                seed = None
            train_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("train", cfg,
                                                                              batch_size,
                                                                              rot_interpol_augmentation=augment,
                                                                              interpolation=interpolation,
                                                                              reshuffle_seed=seed)
            valid_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("valid", cfg,
                                                                              eval_batch_size,
                                                                              rot_interpol_augmentation=False,
                                                                              interpolation=interpolation,
                                                                              reshuffle_seed=seed)
        else:
            train_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("trainval", cfg,
                                                                              batch_size,
                                                                              rot_interpol_augmentation=augment,
                                                                              interpolation=interpolation,
                                                                              reshuffle_seed=None)
            valid_loader = False
        
        test_loader, n_inputs, n_outputs = data_loader_mnist_rot.build_mnist_rot_loader("test", cfg,
                                                                                        eval_batch_size,
                                                                                        rot_interpol_augmentation=False)
     
    elif dataset == "mnist_fliprot":
        
        if validation:
            if reshuffle:
                seed = np.random.randint(0, 100000)
            else:
                seed = None
            
            train_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("train", cfg,
                                                                                  batch_size,
                                                                                  rot_interpol_augmentation=augment,
                                                                                  interpolation=interpolation,
                                                                                  reshuffle_seed=seed)
            valid_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("valid", cfg,
                                                                                  eval_batch_size,
                                                                                  rot_interpol_augmentation=False,
                                                                                  interpolation=interpolation,
                                                                                  reshuffle_seed=seed)
        else:
            train_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("trainval", cfg,
                                                                                  batch_size,
                                                                                  rot_interpol_augmentation=augment,
                                                                                  interpolation=interpolation,
                                                                                  reshuffle_seed=None)
            valid_loader = False
        
        test_loader, n_inputs, n_outputs = data_loader_mnist_fliprot.build_mnist_rot_loader("test", cfg,
                                                                                            eval_batch_size,
                                                                                            rot_interpol_augmentation=False)
    elif dataset == "mnist12k":
        
        if validation:
            if reshuffle:
                seed = np.random.randint(0, 100000)
            else:
                seed = None
            train_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("train", cfg,
                                                                            batch_size,
                                                                            rot_interpol_augmentation=augment,
                                                                            interpolation=interpolation,
                                                                            reshuffle_seed=seed)
            valid_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("valid", cfg,
                                                                            eval_batch_size,
                                                                            rot_interpol_augmentation=False,
                                                                            interpolation=interpolation,
                                                                            reshuffle_seed=seed)
        else:
            train_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("trainval", cfg,
                                                                            batch_size,
                                                                            rot_interpol_augmentation=augment,
                                                                            interpolation=interpolation,
                                                                            reshuffle_seed=None)
            valid_loader = False
        
        test_loader, n_inputs, n_outputs = data_loader_mnist12k.build_mnist12k_loader("test", cfg,
                                                                                      eval_batch_size,
                                                                                      # rot_interpol_augmentation=False
                                                                                      # interpolation=interpolation,
                                                                                      )
    elif dataset == "STL10":
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_stl10.build_stl10_loaders(
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset == "STL10cif":
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_stl10.build_stl10cif_loaders(
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset.startswith("STL10|"):
        size = int(dataset.split("|")[1])
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_stl10frac.build_stl10_frac_loaders(
            size,
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset.startswith("STL10cif|"):
        size = int(dataset.split("|")[1])
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_stl10frac.build_stl10cif_frac_loaders(
            size,
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset == "cifar10":
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_cifar10.build_cifar10_loaders(
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset == "cifar100":
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_cifar100.build_cifar100_loaders(
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset == "imagenette":
        resolution = cfg.training.dataset.resolution or None
        resolution_test = cfg.dataset.resolution_test or None
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_imagenette.build_imagenette_loaders(
            batch_size,
            eval_batch_size,
            augment=augment,
            num_workers=num_workers,
            drop_last=drop_last_train,
            resolution=resolution,
            resolution_test = resolution_test,
        ) 
    elif dataset == "Galaxy10_DECals":
        resolution = cfg.training.dataset.resolution
        dir = cfg.training.dataset.data_dir
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_Galaxy10_DECals.build_galaxy10_loaders(
            batch_size,
            eval_batch_size,
            data_dir=dir,
            augment=augment,
            num_workers=num_workers,
            resolution=resolution,
        ) 

    else:
        raise ValueError("Dataset '{}' not recognized!".format(dataset))
    
    dataloaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

    if n_outputs == 2:
        n_outputs = 1

    return dataloaders, n_inputs, n_outputs


def allowed_usage_time(
    gpu_time_limit: bool,
    start_time: datetime.time = datetime.time(hour=8, minute=30),
    end_time: datetime.time = datetime.time(hour=20),
):
    """
    GPU sharing. Check if the current time is within the allowed usage time.
    """
    if not gpu_time_limit:
        return
    # Get the current time in GMT+2
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).time()

    # Check if the current time is within the range
    if start_time <= now <= end_time:
        raise ValueError("GPU usage not allowed between 8am and 8pm GMT+2")