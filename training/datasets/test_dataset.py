import hydra
from omegaconf import DictConfig
import numpy as np

import sys
import os
sys.path.append(f"{os.getcwd()}")
from training import utils
os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    #print(cfg)
    dataloaders, normalize_weights = hydra.utils.call(cfg.training.dataset)

    train_dataloader = dataloaders["train"]
    valid_dataloader = dataloaders["valid"]
    test_dataloader = dataloaders["test"]

    #print("len(train_dataloader.dataset): ", len(train_dataloader.dataset))
    #print("len(valid_dataloader.dataset): ", len(valid_dataloader.dataset))
    #print("len(test_dataloader.dataset): ", len(test_dataloader.dataset))

    for i, out_dataloader in enumerate(train_dataloader):
        images, labels, meta_data = utils.get_out_dataloader(out_dataloader)
        print(images.shape)
        print(labels.shape)
        break
    

if __name__ == "__main__":
    main()