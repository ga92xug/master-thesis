import hydra
from omegaconf import DictConfig
import numpy as np

import sys
import os
from pandas import value_counts

sys.path.append(f"{os.getcwd()}")
from training import utils
from training.datasets.utils import get_normalize_weights
os.environ['HYDRA_FULL_ERROR'] = '1'

def get_stats(dataloader):
    list_images = []
    list_labels = []
    for i, out_dataloader in enumerate(dataloader):
        #print(i)
        images, labels, meta_data = utils.get_out_dataloader(out_dataloader)
        list_images.append(images.cpu().numpy())
        list_labels.append(labels.cpu().numpy())

    images = np.concatenate(list_images, axis=0)
    mean = np.mean(images, axis=(0, 2, 3)).tolist()
    std = np.std(images, axis=(0, 2, 3)).tolist()
    print("mean", mean)
    print("std", std)

    labels = np.concatenate(list_labels, axis=0)
    print("labels", labels.shape)
    print("value_counts", value_counts(labels))

    #get_normalize_weights(labels)

    return mean, std

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:

    if hasattr(cfg, 'mean_std_test'):
        print("mean_std_test")
        cfg.training.dataset.channel_wise_mean_images = None
        cfg.training.dataset.channel_wise_std_images = None

    #print(cfg)

    dataloaders, normalize_weights = hydra.utils.call(cfg.training.dataset)

    train_dataloader = dataloaders["train"]
    valid_dataloader = dataloaders["valid"]
    test_dataloader = dataloaders["test"]

    length_train = len(train_dataloader.dataset)
    length_valid = len(valid_dataloader.dataset)
    length_test = len(test_dataloader.dataset)
    lenght_all = length_train + length_valid + length_test

    print("train:", length_train, "of all", length_train/lenght_all)
    print("valid:", length_valid, "of all", length_valid/lenght_all)
    print("test:", length_test, "of all", length_test/lenght_all)
    
    for name, dataloader in dataloaders.items():
        print("\nDataloader: ", name)
        #if name != "test":
        #    continue
        for i, out_dataloader in enumerate(dataloader):
            images, labels, meta_data = utils.get_out_dataloader(out_dataloader)
            print(images.shape)
            print(labels.shape)
            break
        
        mean, std = get_stats(dataloader)
        

if __name__ == "__main__":
    main()