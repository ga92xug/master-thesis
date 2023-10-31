import os
from typing import Dict, List
import foolbox as fb
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
    ) -> Dict:

    results = {}
    #model = model.eval()

    fmodel = fb.PyTorchModel(
        model=model, 
        bounds=(0, 1), 
        preprocessing = dict(mean=mean, std=std, axis=-3)
    )

    attacks = [
        fb.attacks.LinfProjectedGradientDescentAttack(), 
        fb.attacks.L2ProjectedGradientDescentAttack(),
        fb.attacks.LinfFastGradientAttack(),
        fb.attacks.LinfBasicIterativeAttack(),
    ]
    epsilons = [0.0, 0.0005, 0.001, 0.01, 0.03, 0.1]

    for attack in attacks:
        attack_name = attack.__class__.__name__
        results[attack_name] = {}
        for epsilon in epsilons:
            print(f"Running {attack_name} attack with epsilon {epsilon}")
            results[attack_name][epsilon] = {}
            total = 0
            count_adv = 0

            for i, out_dataloader in enumerate(dataloader):
                images, labels, _ = utils.get_out_dataloader(out_dataloader, device)
                # denormalize for attack
                images = images * std[:, None, None] + mean[:, None, None]

                _, advs_images, is_adv = attack(fmodel, images, labels, epsilons=epsilon)
                count_adv += is_adv.sum().item()
                total += images.shape[0]

            robust_accuracy = 1 - (count_adv / total)
            results[attack_name][epsilon]["count_adv"] = count_adv
            results[attack_name][epsilon]["robust_acc"] = robust_accuracy

    results["total"] = total
    return results


    

