from pathlib import Path
from typing import Tuple, Callable, Optional
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange
from pickle import load

import sys
sys.path.append('../imagenet') # add parent directory

ROOT_DIR = '../Data/frischs/datasets/'
DATA_DIR = ROOT_DIR + "imagenet/" #cifar-10-batches-py"


class ImageNet(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        version: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        assert version in [32, 64]
        self.transform = transform if transform is not None else lambda x: x
        self.target_tf = (
            target_transform if target_transform is not None else lambda x: x
        )
        root = Path(root)
        version_squared = version * version
        self.imgs, self.labels = None, None
        if train:
            self.imgs, self.labels = [], []
            for i in tqdm(
                range(9), total=10, leave=False, desc=f"Loading ImageNet{version}..."
            ):
                data_dict = unpickle(root / f"train_data_batch_{i+1}")
                img_data = data_dict["data"]
                label_data = data_dict["labels"]
                img_data = img_data / float32(255)
                label_data = [i - 1 for i in label_data]
                img_data = np.dstack(
                    (
                        img_data[:, :version_squared],
                        img_data[:, version_squared : 2 * version_squared],
                        img_data[:, 2 * version_squared :],
                    )
                )
                img_data = img_data.reshape(
                    (img_data.shape[0], version, version, 3)
                )  # .transpose(0, 3, 1, 2)
                self.imgs.append(img_data)
                self.labels.extend(label_data)
            self.imgs = np.concatenate(self.imgs, axis=0)
        else:
            data_dict = unpickle(root / "val_data")
            img_data = data_dict["data"]
            label_data = data_dict["labels"]
            img_data = img_data / float32(255)
            label_data = [i - 1 for i in label_data]
            img_data = np.dstack(
                (
                    img_data[:, :version_squared],
                    img_data[:, version_squared : 2 * version_squared],
                    img_data[:, 2 * version_squared :],
                )
            )
            img_data = img_data.reshape(
                (img_data.shape[0], version, version, 3)
            )  # .transpose(0, 3, 1, 2)
            self.imgs = img_data
            self.labels = label_data
        if normalize:
            mean = np.array((0.485, 0.456, 0.406)).reshape(1, 1, 1, 3)
            std = np.array((0.229, 0.224, 0.225)).reshape(1, 1, 1, 3)
            self.imgs = rearrange(
                torch.Tensor((self.imgs - mean) / std), "B H W C -> B C H W", C=3
            )
    def __len__(self):
        assert self.imgs.shape[0] == len(self.labels)
        return len(self.labels)
    def __getitem__(self, index: int) -> Tuple:
        return self.transform(self.imgs[index]), self.target_tf(self.labels[index]) 


def unpickle(file):
    with open(file, "rb") as open_file:
        data_dict = load(open_file)
    return data_dict


if __name__ == "__main__":
    imagenet = ImageNet(root=DATA_DIR, train=True, version=32)
    print(imagenet[0][0].shape)