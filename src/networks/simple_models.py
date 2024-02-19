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
#from src.utils.utils import compare_state_dicts
from src.networks.eq_nasnet.eq_nasnet import EquivariantNASNet
from equivariant.nn.group_tensor import GroupTensor
from equivariant.nn.modules.conv.r2convolution import R2Conv
from equivariant.nn.modules.restriction_module import RestrictionModule

from src.networks.util import (
    get_group_id, 
    get_gspace_from_id, 
    adjusted_out_channels,
)

from equivariant.nn.field_type import FieldType
from src.networks.eq_convs import EquivariantConv
from src.networks.eq_restriction import Restriction, Restriction_Group_or_CNN, Restriction_from_id
from equivariant.nn import (
    GroupTensor,
    FieldType,
    BatchNorm,
    Mish,
    ReLU,
    Swish,
)

def hash_tensor(tensor):
    return hash(tuple(tensor.reshape(-1).tolist()))


class Simple_Eq_Net_restriction(nn.Module):
    def __init__(self):
        super(Simple_Eq_Net_restriction, self).__init__()  # Corrected: Call to the superclass constructor
        gspace = get_gspace_from_id((0, 4))
        self.in_type = FieldType(
                gspace, [gspace.trivial_repr] * 3
            )
        self.out_type = FieldType(
            self.in_type.gspace,
            [self.in_type.gspace.regular_repr] * 16,
        )
        
        self.conv = R2Conv(
            self.in_type,
            self.out_type,
            kernel_size=5,
            padding=2,
            stride=2,
            dilation=1,
            groups=1,
            bias=False,
            sigma=None,
            frequencies_cutoff=lambda r: 3 * r,
        )
        self.restrict = Restriction_from_id(self.out_type, (-1, 4))
        self.conv1 = R2Conv(
            self.restrict.out_type,
            FieldType(self.in_type.gspace,[self.in_type.gspace.regular_repr] * 16,),
            kernel_size=5,
            padding=2,
            stride=2,
            dilation=1,
            groups=1,
            bias=False,
            sigma=None,
            frequencies_cutoff=lambda r: 3 * r,
        )


    def forward(self, x):
        x = GroupTensor(x, self.in_type)
        x = self.conv(x)
        #print("conv", x.tensor.view(-1).data)
        print("conv hash", hash_tensor(x.tensor))
        x = self.restrict(x)
        #print("restrict hash", x.tensor.sum().item())
        
        x = self.conv1(x)
        #print("conv1 hash", hash_tensor(x.tensor))
        return x.tensor

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        super().load_state_dict(state_dict, strict)
        
        for name, layer in self.named_modules():
            if isinstance(layer, R2Conv):
                _filter, _bias = layer.expand_parameters()
                layer.filter = _filter
                if _bias is not None:
                    layer.expanded_bias = _bias
                else:
                    layer.expanded_bias = None


class Simple_Eq_Net(nn.Module):
    def __init__(self):
        super(Simple_Eq_Net, self).__init__()  # Corrected: Call to the superclass constructor
        gspace = get_gspace_from_id((0, 4))
        self.in_type = FieldType(
                gspace, [gspace.trivial_repr] * 3
            )
        self.out_type = FieldType(
            self.in_type.gspace,
            [self.in_type.gspace.regular_repr] * 16,
        )
        
        self.conv = R2Conv(
            self.in_type,
            self.out_type,
            kernel_size=5,
            padding=2,
            stride=2,
            dilation=1,
            groups=1,
            bias=False,
            sigma=None,
            frequencies_cutoff=lambda r: 3 * r,
        )


    def forward(self, x):
        x = GroupTensor(x, self.in_type)
        x = self.conv(x)
        return x.tensor


    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        super().load_state_dict(state_dict, strict)
        
        for name, layer in self.named_modules():
            if isinstance(layer, R2Conv):
                layer.expand_parameters()
                
                