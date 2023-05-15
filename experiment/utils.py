import os.path
import sqlite3
import pandas as pd
import numpy as np
import io
import datetime

from typing import List

#from models import *
from networks import *

# the values of these command line arguments are used to define the name of the experiments
# you can add more names in this list
EXPERIMENT_PARAMETERS = ["model", "type", "N", "flip", "restrict", "sgsize", "fixparams", "augment", "F", "sigma", "interpolation"]


def allowed_usage_time(
    start_time: datetime.time = datetime.time(hour=8, minute=30),
    end_time: datetime.time = datetime.time(hour=20),
):
    """
    GPU sharing. Check if the current time is within the allowed usage time.
    """
    # Get the current time in GMT+2
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).time()

    # Check if the current time is within the range
    if start_time <= now <= end_time:
        raise ValueError("GPU usage not allowed between 8am and 8pm GMT+2")


########################################################################################################################
# Utilites to build paths and names in a standard way
########################################################################################################################

def exp_name(cfg):
    values = []
    # model name
    if cfg.model._target_ == "networks.EquivariantWideResNet":
        values.append("eq_wrn")
    elif cfg.model._target_ == "networks.EquivariantResNet9":
        values.append("res9")
    elif cfg.model._target_ == "networks.EquivariantMobileNetV2":
        values.append("eq_mobv2")
    elif cfg.model._target_ == "networks.RandomNet":
        values.append("rand")
    elif cfg.model._target_ == "networks.WideResNet":
        values.append("wrn")
    else:
        ValueError("Unknown model")

    # depth
    if cfg.model._target_ in ["networks.WideResNet", "networks.EquivariantWideResNet"]:
        values.append(f"{cfg.model.depth}")
    elif cfg.model._target_ in ["networks.EquivariantMobileNetV2"]:
        values.append(f"{cfg.model.depth_multiplier}")
    
    # width
    if cfg.model._target_ in ["networks.WideResNet", "networks.EquivariantWideResNet"]:
        values.append(f"{cfg.model.widen_factor}")
    elif cfg.model._target_ in ["networks.EquivariantMobileNetV2"]:
        values.append(f"{cfg.model.width_multiplier}")
    
    # kernel size
    if cfg.model._target_ in ["networks.WideResNet", "networks.EquivariantWideResNet"]:
        if len(cfg.model.kernel_layout) == 3:
            values.append(f"B({cfg.model.kernel_layout[0]},{cfg.model.kernel_layout[1]},{cfg.model.kernel_layout[2]})")
        elif len(cfg.model.kernel_layout) == 2:
            values.append(f"B({cfg.model.kernel_layout[0]},{cfg.model.kernel_layout[1]})")
        elif len(cfg.model.kernel_layout) == 1:
            values.append(f"B({cfg.model.kernel_layout[0]})")
        elif len(cfg.model.kernel_layout) == 4:
            values.append(f"B({cfg.model.kernel_layout[0]},{cfg.model.kernel_layout[1]},{cfg.model.kernel_layout[2]},{cfg.model.kernel_layout[3]})")

    if cfg.model._target_.startswith("eq_"):
        values.append(f"rot{cfg.model.rotation}")
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
from experiment.datasets.STL10 import data_loader_stl10
from experiment.datasets.STL10 import data_loader_stl10frac
from experiment.datasets.imagenette import data_loader_imagenette


def build_dataloaders(cfg):
    dataset = cfg.dataset.name
    batch_size = cfg.training.batch_size
    num_workers = cfg.dataset.workers
    drop_last_train = cfg.dataset.drop_last_train or False
    augment = cfg.dataset.augment or False
    validation = cfg.training.earlystop or True 
    reshuffle = cfg.dataset.reshuffle or False
    eval_batch_size = cfg.training.eval_batch_size or None
    interpolation = cfg.dataset.interpolation or 2
    
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
        resolution = cfg.dataset.resolution or None
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
    else:
        raise ValueError("Dataset '{}' not recognized!".format(dataset))
    
    dataloaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    return dataloaders, n_inputs, n_outputs

