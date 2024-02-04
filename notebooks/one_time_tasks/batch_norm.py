from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import hydra
import lightning as L
from lightning.pytorch.strategies import DDPStrategy

import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from lightning.pytorch.callbacks import BasePredictionWriter
from torch import nn
import os


os.environ['HYDRA_FULL_ERROR'] = '1'

import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

from src.data.datamodule import DataModule
from src.training_loop.lightning_module import LitModule
from src.utils.ckpt_path import get_ckpt_path
from src.networks.eq_nasnet.eq_nasnet import EquivariantNASNet
from equivariant.nn.group_tensor import GroupTensor
from equivariant.nn.modules.conv.r2convolution import R2Conv
from equivariant.nn.modules.restriction_module import RestrictionModule
from equivariant.nn.modules.utils import indexes_from_labels


from src.networks.util import (
    get_group_id, 
    get_gspace_from_id, 
    adjusted_out_channels,
)

from equivariant.nn.field_type import FieldType
from src.networks.eq_convs import EquivariantConv
from src.networks.eq_restriction import Restriction, Restriction_Group_or_CNN, Restriction_from_id
from equivariant.nn import (
    BatchNorm,
    GroupTensor,
    FieldType,
    BatchNorm,
    Mish,
    ReLU,
    Swish,
)

def print_gpu_memory_usage(label=""):
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"{label} - Memory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB")


def batch_norm_test():
    gspace = get_gspace_from_id((0, 4))
    num_represntations = 100
    in_type = FieldType(
            gspace, [gspace.regular_repr] * num_represntations
        )
    out_type = FieldType(
        in_type.gspace,
        [in_type.gspace.regular_repr] * num_represntations,
    )

    grouped_fields = indexes_from_labels(
            in_type, [r.size for r in in_type.representations]
        )
    
    print_gpu_memory_usage("Before BatchNorm")
    batch_norm = BatchNorm(in_type, affine=True, eps=1e-05, momentum=0.1, track_running_stats=True).cuda()

    # forward
    input = torch.rand(1, 8 * num_represntations, 224, 224).cuda()
    input = GroupTensor(input, in_type)
    output = batch_norm(input)
    print_gpu_memory_usage("After BatchNorm")


if __name__ == "__main__":
    batch_norm_test()