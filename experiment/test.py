"""
Test for the dataloader.
Had an issue with drop_last=True for train but not for valid/test.
"""

import hydra

import sys
sys.path.append('../scaling-laws-ecnn/') # add parent directory

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def __main__(cfg):
    import utils
    
    
    _dataloaders, n_inputs, n_outputs = utils.build_dataloaders(cfg)
    train_loader = _dataloaders['train']
    train_len = len(train_loader)
    data_len = len(train_loader.dataset)
    print("Train loader length:", train_len)
    print("Train dataset length:", data_len)

    n_samples = 0
    for i, (images, labels) in enumerate(_dataloaders['train']):
        n_samples += images.shape[0]
        
    print("Number of samples:", n_samples, data_len)

if __name__ == "__main__":
    __main__()