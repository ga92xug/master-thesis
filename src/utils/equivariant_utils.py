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
        

def update_filters_with_pretrained_weights(net: nn.Module):
    """Update the filters and bias of the equivariant layers with the pretrained weights."""
    # the filter and bias have to be recomputed with the loaded weights
    for name, layer in net.named_modules():
        if isinstance(layer, R2Conv):
            _filter, _bias = layer.expand_parameters()
            layer.filter = _filter
            if _bias is not None:
                layer.expanded_bias = _bias
            else:
                layer.expanded_bias = None