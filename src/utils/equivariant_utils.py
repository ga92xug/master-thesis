from typing import Tuple
from torch import nn

from equivariant.nn.modules.equivariant_module import EquivariantModule
from equivariant.nn.modules.conv import R2Conv

def is_equivariant_model(network: nn.Module) -> bool:
    """Check if the model is equivariant."""
    for name, module in network.named_modules():
        if isinstance(module, EquivariantModule):
            return True
        
    return False
        

def create_filters_network(net: nn.Module):
    """Update the filters and bias of the equivariant layers with the pretrained weights."""
    # the filter and bias have to be recomputed with the loaded weights
    for name, layer in net.named_modules():
        if isinstance(layer, R2Conv):
            print(name)
            layer.expand_parameters()