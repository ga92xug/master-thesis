import copy
import os
import hashlib
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig
import torch.optim as optim
import lightning as L

import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
from src.data.datamodule import DataModule
from src.networks.simple_models import Simple_Eq_Net
from src.save_load.utils import *
from src.training_loop.lightning_module import LitModule

os.environ['HYDRA_FULL_ERROR'] = '1'


def forward_backward_pass(net, input_data):
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss_fn = torch.nn.CrossEntropyLoss()
    net.train()
    for name, module in net.named_modules():
        if "dropout" in name:
            module.p = 0.0
    net.zero_grad()
    output = net(input_data)
    target = torch.ones_like(output)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss



def train(net, steps, input_data, saving=False):
    output_list = []
    grad_list = []
    output_saver = ModelOutputSaver()
    attach_hooks(net, output_saver)

    # Perform x number of update steps with SGD and save outputs and gradients
    
    for i in range(steps):
        print("Step", i)
        loss = forward_backward_pass(net, input_data)
        output, grad = copy.deepcopy(output_saver.outputs), copy.deepcopy(output_saver.grads)
        #print("loss", loss.item())

        if not saving:
            output_list.append(output)
            grad_list.append(grad)
            continue
        else:
            if i == 0:
                #print("Saving model state dict")
                #print("hash weights", hash_tensor(net.conv.weights))
                save_model_state_dict(net)

            if i >= steps // 2:
                #for layer_name, grad in output_before_update.items():
                output_list.append(output) 
                grad_list.append(grad)
                #output_tensor = torch.cat((output_tensor, output.unsqueeze(0)), dim=0)
                #grad_tensor = torch.cat((grad_tensor, grad_before_update.unsqueeze(0)), dim=0)
            
    return output_list, grad_list


def normal_case(cfg):
    image_size = cfg.train.dataset.resolution
    steps = 3
    input_data = torch.randn(1, 3, image_size, image_size)
    net = get_model(cfg)

    # Initial forward-backward pass
    #loss = forward_backward_pass(net, input_data)
    #print("output_before_update", output_before_update)
    #print("grad_before_update", grads)
    #print("loss", loss)
    #quit()
    output_list, grad_list = train(net, steps, input_data, saving=True)
    # Save outputs and gradients using torch.save
    save_outputs_and_grads(output_list, grad_list, input_data)

    print("\nNew model:")
    new_run_comparison(cfg)

def save_outputs_and_grads(outputs_tensor, grads_tensor, input_data):
    path = "pre_trained_models/test/outputs_grads.pth"

    torch.save({
        "layer_outputs": outputs_tensor,
        "layer_gradients": grads_tensor,
        "input_data": input_data
    }, path)

def save_model_state_dict(net):
    path = "pre_trained_models/test/model_state_dict.pth"
    torch.save(net.state_dict(), path)

def new_run_comparison(cfg):
    # Load saved model state dictionary
    saved_model_state_dict = torch.load("pre_trained_models/test/model_state_dict.pth")
    
    # Instantiate new model and load saved model state dictionary
    new_net = get_model(cfg)
    new_net.load_state_dict(saved_model_state_dict)
    
    # Load saved outputs and gradients
    saved_data = torch.load("pre_trained_models/test/outputs_grads.pth")
    saved_outputs = saved_data["layer_outputs"]
    saved_grads = saved_data["layer_gradients"]
    saved_input = saved_data["input_data"]

    steps = len(saved_outputs)
    output_list, grad_list = train(new_net, steps, saved_input, saving=False)

    print("\nComparing outputs and gradients:")
    for i in range(steps):
        print(f"Step {i}")
        discrepancies = compare_model_outputs(saved_outputs[i], output_list[i])
        if len(discrepancies) > 0:
            print(f"Discrepancies found in outputs {i}")
            break

        discrepancies = compare_gradients(saved_grads[i], grad_list[i])
        if len(discrepancies) > 0:
            print(f"Discrepancies found in gradients {i}")
            break

    print("\nOverall no discrepancies found! The models are equivalent")
    

@hydra.main(version_base="1.3", config_path="../../configs", config_name="conf.yaml")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.manual_seed(cfg.seed)
    if cfg.get("train_mode"):
        normal_case(cfg)
    else:
        new_run_comparison(cfg)

if __name__ == "__main__":
    main()
