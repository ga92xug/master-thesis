from pathlib import Path
import torch
from typing import Dict, List, Tuple
import torch
import sys
import os
import re
import time
#from autoattack import AutoAttack

sys.path.append(f"{os.getcwd()}")
#from training import utils


def run_autoattack(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        mean: torch.Tensor,
        std: torch.Tensor,
        verbose: int,
        num_classes: int,
    )-> Dict:
    global_start_time = time.time()

    # get results
    aa_state = _AutoAttackState(global_start_time)
    
    adversary = configure_attack(
        model=model,
        aa_state_path=aa_state.path,
        num_classes=num_classes,
        verbose=verbose,
    )

    images, labels = preprocess_images(
        dataloader=dataloader,
        device=device,
        mean=mean,
        std=std,
    )

    # run attack
    adversarial_images = adversary.run_standard_evaluation(images, labels)
    results = aa_state.extract_percentages()
    aa_state.remove_file()

    return results

def preprocess_images(
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        mean: List[float],
        std: List[float],
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess images from dataloader.
    """

    list_images = []
    list_labels = []
    for i, out_dataloader in enumerate(dataloader):
        images, labels = next(out_dataloader)
        #images, labels, meta_data = utils.get_out_dataloader(out_dataloader, device)
        if normalize:
            images = images * std[:, None, None] + mean[:, None, None]
        list_images.append(images)
        list_labels.append(labels)

    images = torch.cat(list_images, dim=0)
    labels = torch.cat(list_labels, dim=0)
    return images, labels

class _AutoAttackState:
    def __init__(self, global_start_time, location="/home/frischs/aa_state"):
        self.global_start_time = global_start_time
        self.location = location
        self.path = self.create_file()

    def create_file(self) -> Path:
        if not os.path.exists(self.location):
            os.makedirs(self.location)
        
        file_name = f"{self.global_start_time}.txt"
        aa_state_path = os.path.expanduser(os.path.join(self.location, file_name))
        aa_state_path = Path(aa_state_path)
        
        if aa_state_path.exists():
            os.remove(aa_state_path)
        
        return aa_state_path

    def remove_file(self):
        if self.path.exists():
            os.remove(self.path)

    def extract_percentages(self):
        with open(self.path, 'r') as f:
            content = f.read()

        result_dict = {}
        methods = ['APGD-CE', 'APGD-T', 'FAB-T', 'initial accuracy', 'robust accuracy']

        for method in methods:
            pattern = f"{method}: ([0-9]*\.?[0-9]+)%"
            match = re.search(pattern, content)
            
            if match:
                result_dict[method] = float(match.group(1))

        # are in % for logging we want them in [0,1]
        result_dict = {k: v / 100 for k, v in result_dict.items()}                
        return result_dict


def configure_attack(
        model: torch.nn.Module,
        aa_state_path: Path,
        num_classes: int,
        version: str = 'standard', 
        verbose: bool = False,
    ) -> AutoAttack:
    adversary = AutoAttack(
        model=model, 
        norm='Linf', 
        eps=8/255, 
        log_path=aa_state_path,
        verbose=verbose,
    )

    if version == 'standard':
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        if adversary.norm in ['Linf', 'L2']:
            adversary.apgd.n_restarts = 1
            adversary.apgd_targeted.n_target_classes = num_classes - 1 # 9
        elif adversary.norm in ['L1']:
            adversary.apgd.use_largereps = True
            adversary.apgd_targeted.use_largereps = True
            adversary.apgd.n_restarts = 5
            adversary.apgd_targeted.n_target_classes = num_classes -1 # 5
        adversary.fab.n_restarts = 1
        adversary.apgd_targeted.n_restarts = 1
        adversary.fab.n_target_classes = num_classes - 1 # 9
        #adversary.apgd_targeted.n_target_classes = 9
        adversary.square.n_queries = 5000

    return adversary