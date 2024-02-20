import os
import hashlib
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig
import torch.optim as optim
from equivariant.nn.group_tensor import GroupTensor
from equivariant.nn.modules.conv.basisexpansion import BasisExpansion
from equivariant.nn.modules.conv.r2convolution import R2Conv

from src.data.datamodule import DataModule
from src.networks.compare_models.compare_models import EfficientNet
from src.networks.simple_models import *
from src.save_load.forward import hash_tensor
from src.training_loop.lightning_module import LitModule


def get_model(cfg):
    image_size = cfg.train.dataset.resolution
    #net = More_Complex_CNN()
    #net = Simple_Eq_Net()
    #net = Simple_CNN()
    #net = Simple_Eq_Net_restriction()
    net = EquivariantNASNet(blocks_args_dict=cfg.train.network.blocks_args_dict, image_size=image_size)
    net.eval()
    return net

class ModelOutputSaver:
    def __init__(self):
        self.outputs = {}
        self.grads = {}

    def save_output(self, layer_name):
        def hook(module, input, output):
            if isinstance(output, GroupTensor):
                output = output.tensor
            if isinstance(output, torch.Tensor):
                self.outputs[layer_name] = output.detach()
            
            #print("output", hash_tensor(output), output.shape)
        return hook

    def save_grad(self, layer_name):
        def hook(module, grad_input, grad_output):
            grad_output = grad_output[0]
            if isinstance(grad_output, GroupTensor):
                grad_output = grad_output.tensor

            if isinstance(grad_output, torch.Tensor):
                self.grads[layer_name] = grad_output
        return hook

def attach_hooks(model, saver):
    for name, layer in model.named_modules():
        #if isinstance(layer, R2Conv) or isinstance(layer, BasisExpansion):
        layer.register_forward_hook(saver.save_output(name))
            
        #if isinstance(layer, R2Conv) or isinstance(layer, BasisExpansion):
        layer.register_full_backward_hook(saver.save_grad(name))


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
            #print(f"Sum discrepancy: {torch.sum(torch.abs(output - saved_output))}")
            print("output", hash_tensor(output), "saved_output", hash_tensor(saved_output))
            break
        else:
            #print(f"No discrepancy in layer {layer_name}")
            pass

    #if len(discrepancies) == 0:
    #    print("No discrepancies found in outputs")
    return discrepancies


def compare_gradients(saved_grads, new_grads):
    discrepancies = {}
    for layer_name, grad in new_grads.items():
        saved_grad = saved_grads.get(layer_name)
        if saved_grad is None:
            raise ValueError(f"Layer {layer_name} not found in saved gradients")
        
        if not torch.equal(grad, saved_grad):
            discrepancies[layer_name] = (grad, saved_grad)
            print(f"Discrepancy in gradients of layer {layer_name}")
    if len(discrepancies) == 0:
        print("No discrepancies found in gradients")
    return discrepancies