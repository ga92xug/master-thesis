import os
from pathlib import Path
import re
from autoattack import AutoAttack
import torch

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