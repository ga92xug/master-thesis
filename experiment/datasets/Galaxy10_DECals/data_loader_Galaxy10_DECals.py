import torch
from tqdm import tqdm
import h5py
import numpy as np

MEAN = [0.01443392, 0.01443392, 0.01443392]
STD = [0.12014125, 0.1125077, 0.10281154]

class Galaxy10Dataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir

        with h5py.File(self.dir, 'r') as F:
            self.images = np.array(F['images'])
            self.labels = np.array(F['ans'])

        self.labels = to_categorical(self.labels, 10)
        self.images = self.images.astype(np.float32)
        self.images = np.transpose(self.images, (0, 3, 1, 2))
        # print("mean: ", np.mean(self.images / 255, axis=(0, 2, 3)))
        # print("std: ", np.std(self.images / 255, axis=(0, 2, 3)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def get_dataloader(
        batch_size,
        eval_batchsize,
        dir,
        resolution,
        num_workers=8,
        augment=False,
        ):
    dataset = Galaxy10Dataset(dir)
    np.random.seed(42)

    # Split the dataset into train, val, and test sets.
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(len(dataset) * 0.8)
    train_indices = indices[:split]
    val_indices = indices[split:split + int(len(dataset) * 0.1)]
    test_indices = indices[split + int(len(dataset) * 0.1):]

    # Create the train, val, and test datasets.
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    valid_transform = torch.utils.data.transforms.Compose([
                torch.utils.data.transforms.ToTensor(),
                torch.utils.data.transforms.Normalize(MEAN, STD),
            ])

    if augment:
        train_transform = torch.utils.data.transforms.Compose([
                torch.utils.data.transforms.ToTensor(),
                torch.utils.data.transforms.Normalize(MEAN, STD),
            ])
    else:
        train_transform = valid_transform

    # Create the train, val, and test dataloaders.
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=train_transform,
        )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batchsize,
        shuffle=False,
        num_workers=num_workers,
        transform=valid_transform,
        )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batchsize,
        shuffle=False,
        num_workers=num_workers,
        transform=valid_transform,
        )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(batch_size=128, eval_batchsize=16, dir=ROOT_DIR + DATASET_FOLDER + DATASET_NAME)

    for i, (images, labels) in enumerate(train_dataloader):
        print(images.shape)
        print(labels.shape)
        break




