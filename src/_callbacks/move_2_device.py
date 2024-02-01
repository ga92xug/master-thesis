from lightning import Callback
from equivariant.nn.modules.conv import R2Conv

class Move_2_Device(Callback):
    def __init__(self):
        super().__init__()

    def setup(self, trainer, pl_module, stage):
        for name, layer in pl_module.net.named_modules():
            if isinstance(layer, R2Conv):
                if not hasattr(layer, "filter"):
                    continue
                device = layer.weights.device
                layer.filter = layer.filter.to(device)
                if layer.expanded_bias is not None:
                    layer.expanded_bias = layer.expanded_bias.to(self.weights.device)
