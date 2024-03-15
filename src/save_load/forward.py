from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import lightning as L
from lightning.pytorch.strategies import DDPStrategy

import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from lightning.pytorch.callbacks import BasePredictionWriter
from torch import Tensor, nn
import os
import hashlib

os.environ['HYDRA_FULL_ERROR'] = '1'
#os.environ['PYTHONHASHSEED'] = '0'

#print("hash", hash("torch"))
#quit()

import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

from src.data.datamodule import DataModule
from src.training_loop.lightning_module import LitModule
#from src.utils.ckpt_path import get_ckpt_path
#from src.utils.utils import compare_state_dicts
from src.networks.eq_nasnet.eq_nasnet import EquivariantNASNet
from equivariant.nn.group_tensor import GroupTensor
from equivariant.nn.modules.conv.r2convolution import R2Conv
from equivariant.nn.modules.restriction_module import RestrictionModule
from src.networks.simple_models import Simple_Eq_Net, Simple_Eq_Net_restriction


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

def hash_tensor(x: Tensor) -> Tensor:
    #assert x.dtype == torch.int64
    while x.ndim > 0:
        x = _reduce_last_axis(x)
    return x.item()

@torch.no_grad()
def _reduce_last_axis(x: Tensor) -> Tensor:
    #assert x.dtype == torch.int64
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        acc *= 6.3
        acc += 1
        acc += x[..., i]
        acc %= 2**32  # Not really necessary.
    return acc

class ModelOutputSaver:
    def __init__(self):
        self.outputs = {}

    def save_output(self, layer_name):
        def hook(module, input, output):
            if isinstance(output, GroupTensor):
                output = output.tensor
            self.outputs[layer_name] = output.detach()
        return hook

def attach_hooks(model, saver):
    for name, layer in model.named_modules():
        if not list(layer.children()):
            # Attach hook to all layers you are interested in (e.g., skip ReLU, BatchNorm, etc.)
            #if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):  # Customize as needed
            layer.register_forward_hook(saver.save_output(name))

def hash_of_module(net, module: str):
    #_conv_stem.conv2d.conv._basisexpansion
    for name, m in net.named_modules():
        if name == module:
            print("hash of module", name)
            return hash(name)
    
    raise ValueError(f"Module {module} not found in net")

def hash_basis_expansion(net):
    _hash = hashlib.sha256()
    #for name, module in net.named_modules():
    #    print("name", name)
    #module = net._conv_stem.conv2d.conv._basisexpansion
    module = net.conv._basisexpansion


    for io in module._representations_pairs:
        #print("module.sampled_bases", hash_tensor(torch.sum(module.sampled_bases[io], dim=(3))), hash_tensor(module.sampled_bases[io]))
        print("module.sampled_bases", hash_tensor(module.sampled_bases[io]))
        _hash.update(str(module.sampled_bases[io]).encode("utf-8"))
    return int(str(int(_hash.hexdigest(), 16))[:5])


def forward_pass(net, input_data):
    # Instantiate ModelOutputSaver
    output_saver = ModelOutputSaver()

    # Attach hooks to model
    attach_hooks(net, output_saver)

    # Forward pass
    net.train()
    for name, module in net.named_modules():
        if "dropout" in name:
            module.p = 0.0
    net(input_data)
    net.eval()
    #with torch.no_grad():
    #    net(input_data)

    net.eval()
    return output_saver.outputs

def hash_tensor_2(tensor):
    _hash = hashlib.sha256()
    _hash.update(str(tuple(tensor.reshape(-1).tolist())).encode("utf-8"))
    return _hash.hexdigest()[:7]

def normal_case(cfg):
    image_size = 2 #cfg.train.dataset.resolution
    # Prepare a sample input data
    input_data = torch.randn(1, 3, image_size, image_size)  # Adjust shape as per your model's requirement

    # inital model
    net = get_model(cfg)  
     
    output_before_save = forward_pass(net, input_data)

    #hash_basis = hash_basis_expansion(net) 

    print()
    #print(len(net.conv._basisexpansion.sampled_bases))
    for key, value in net.named_modules():
        print("key", key)

    for key, value in net.named_buffers(remove_duplicate=False):
        print("key", key, "value", hash_tensor_2(value))
    #print("hash_basis", hash_basis)   
    # Save model and outputs
    path = "pre_trained_models/model_and_outputs.pth"
    net.eval()
    torch.save({
        "model_state_dict": net.state_dict(),
        "layer_outputs": output_before_save,
        "input_data": input_data 
    }, path)
    quit()

    # load new model
    print()
    L.seed_everything(cfg.seed + 1)
    saved_data = torch.load(path)
    model_state_dict = saved_data["model_state_dict"]
    new_net = get_model(cfg) 
    hash_basis = hash_basis_expansion(net) 
    quit()
    new_net.load_state_dict(model_state_dict)

    hash_basis = hash_basis_expansion(net)  
    print("hash_basis", hash_basis)   

    output_after_save = forward_pass(new_net, input_data)

    print()

    # compare
    discrepancies = compare_model_outputs(saved_data["layer_outputs"], output_after_save)
    #print("Discrepancies", discrepancies)


def compare_model_outputs(saved_outputs, new_outputs):
    # Compare outputs with saved outputs
    discrepancies = {}
    for layer_name, output in new_outputs.items():
        saved_output = saved_outputs.get(layer_name)
        if saved_output is None:
            raise ValueError(f"Layer {layer_name} not found in saved outputs")
        if not torch.equal(output, saved_output):
            discrepancies[layer_name] = (output, saved_output)
            print(f"Discrepancy in layer {layer_name}:")
            print(f"Sum discrepancy: {torch.sum(torch.abs(output - saved_output))}")
        else:
            #print(f"No discrepancy in layer {layer_name}")
            pass

    if len(discrepancies) == 0:
        print("No discrepancies found")
    return discrepancies

def new_run_comparison(cfg, path="pre_trained_models/model_and_outputs.pth"):
    image_size = cfg.train.dataset.resolution
    saved_data = torch.load(path)
    model_state_dict = saved_data["model_state_dict"]
    saved_outputs = saved_data["layer_outputs"]
    saved_input = saved_data["input_data"]

    new_net = get_model(cfg)
    #hash_basis = hash_basis_expansion(new_net)
    #print("hash_basis", hash_basis) 
    new_net.load_state_dict(model_state_dict)
      

    output_new_run = forward_pass(new_net, saved_input)

    compare_model_outputs(saved_outputs, output_new_run)
    #print("Discrepancies in new run:", discrepancies)

    return #discrepancies

def get_model(cfg):
    image_size = cfg.train.dataset.resolution
    net = Simple_Eq_Net()
    #net = Simple_Eq_Net_restriction()
    #net = EquivariantNASNet(blocks_args_dict=cfg.train.network.blocks_args_dict, image_size=image_size)
    net.eval()
    return net


@hydra.main(version_base="1.3", config_path="../configs", config_name="conf.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    L.seed_everything(cfg.seed)
    #L.seed_everything(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run the normal case and save outputs
    if cfg.get("train_mode"):
        normal_case(cfg)

    # Run the new run and compare outputs
    if not cfg.get("train_mode"):
        #L.seed_everything(cfg.seed + 1)
        new_run_comparison(cfg)


if __name__ == "__main__":
    main()
