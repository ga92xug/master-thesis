import torch
from tqdm import tqdm
import h5py
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


MEAN = [0.01443392, 0.01443392, 0.01443392]
STD = [0.12014125, 0.1125077, 0.10281154]

class Galaxy10Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.int64)

        return image, label

    
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def build_galaxy10_loaders(
        batch_size,
        eval_batchsize,
        dir,
        resolution,
        num_workers=8,
        augment=False,
        ):
    
    print("Galaxy10 dataset", "batch size", batch_size, "eval batch size", eval_batchsize)

    with h5py.File(dir, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    #labels = to_categorical(labels, 10)
    images = images.astype(np.uint8)

    # Split the data into train, val, and test arrays.
    np.random.seed(42)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    split = int(len(images) * 0.8)
    train_indices = indices[:split]
    val_indices = indices[split:split + int(len(images) * 0.1)]
    test_indices = indices[split + int(len(images) * 0.1):]

    train_images = images[train_indices]
    train_labels = labels[train_indices]
    val_images = images[val_indices]
    val_labels = labels[val_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]

    valid_transform = transforms.Compose([
                transforms.Resize(resolution),
                #transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    if augment:
        train_transform = transforms.Compose([
                transforms.Resize(resolution),
                #transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
    else:
        train_transform = valid_transform

    # Create the train, val, and test datasets.
    train_dataset = Galaxy10Dataset(train_images, train_labels, transform=train_transform)
    val_dataset = Galaxy10Dataset(val_images, val_labels, transform=valid_transform)
    test_dataset = Galaxy10Dataset(test_images, test_labels, transform=valid_transform)


    # Create the train, val, and test dataloaders.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batchsize,
        shuffle=False,
        num_workers=num_workers,
        )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batchsize,
        shuffle=False,
        num_workers=num_workers,
        )

    n_inputs = 3
    n_classes = 10

    return train_dataloader, val_dataloader, test_dataloader, n_inputs, n_classes


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader, n_inputs, n_classes = build_galaxy10_loaders(batch_size=128, eval_batchsize=16, dir= "../Data/frischs/datasets/Galaxy10_DECals/Galaxy10_DECals.h5", resolution=224, num_workers=8, augment=False)

    for i, (images, labels) in enumerate(train_dataloader):
        print(images.shape)
        print(labels.shape)
        print(labels.dtype)

        break




