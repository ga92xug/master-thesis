import os
from typing import Dict, List
import foolbox as fb
import numpy as np
import torch

import sys
import os

sys.path.append(f"{os.getcwd()}")
from training import utils


def foolbox_attack(
        model: torch.nn.Module,
        #images: torch.Tensor,
        #labels: torch.Tensor,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        mean: torch.Tensor,
        std: torch.Tensor,
        sanity_check: bool = False,
    ) -> Dict:

    results = {}
    #model = model.eval()

    fmodel = fb.PyTorchModel(
        model=model, 
        bounds=(0, 1), 
        preprocessing = dict(mean=mean, std=std, axis=-3)
    )

    attacks = [
        # not in use
        #fb.attacks.LinfFastGradientAttack(), # epsilons go up after inital drop

        # used
        #fb.attacks.LinfProjectedGradientDescentAttack(), 
        #fb.attacks.L2ProjectedGradientDescentAttack(),
        #fb.attacks.LinfBasicIterativeAttack(),

        #fb.attacks.L2BasicIterativeAttack(), # potential candidate
        fb.attacks.LinfDeepFoolAttack(), # potential candidate
        fb.attacks.L2DeepFoolAttack(),
        #fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack(), # potential candidate
    ]


    epsilons = np.linspace(0.0, 0.1, num=20)
    #epsilons = [0.0, 0.0005, 0.001, 0.01, 0.03, 0.1]

    for attack in attacks:
        attack_name = attack.__class__.__name__
        results[attack_name] = {}
        total = 0
        count_adv = torch.zeros(len(epsilons)).to(device)
        correct = torch.zeros(len(epsilons)).to(device)


        for i, out_dataloader in enumerate(dataloader):
            images, labels, _ = utils.get_out_dataloader(out_dataloader, device)
            # denormalize for attack
            images = images * std[:, None, None] + mean[:, None, None]

            _, advs_list, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
            count_adv += is_adv.sum(dim=1)
            total += images.shape[0]

            if sanity_check:
                for i, advs in enumerate(advs_list):
                    # normalize for model
                    advs_images = (advs - mean[:, None, None]) / std[:, None, None]
                    preds = model(advs_images).argmax(dim=1)
                    correct[i] += (preds == labels).sum().item()  # Compute accuracy for each epsilon


        robust_accuracies = 1 - (count_adv / total)
        results[attack_name] = [robust_accuracies.tolist(), epsilons.tolist()]
        #for i, epsilon in enumerate(epsilons):
            #results[attack_name][epsilon] = {
            #    "count_adv": count_adv[i].item(),
            #    "robust_acc": robust_accuracies[i].item(),
            #}
            
            
            #print(f"Epsilon {epsilon} Robust accuracy: {robust_accuracies[i]}")
            
        if sanity_check:
            print(f"Accuracy vector for each epsilon: {correct / total}")
            
    #results["total"] = total
    return results


def min_max_images(images: torch.tensor):
    min_ = images.min().item()
    max_ = images.max().item()
    print(f"min: {min_}, max: {max_}")


