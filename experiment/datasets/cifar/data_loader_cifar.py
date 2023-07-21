
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

#import sys
#sys.path.append('../cifar10') # add parent directory

from .autoaugment import CIFAR10Policy

MEAN = np.array([125.3, 123.0, 113.9]) / 255.0  # = np.array([0.49137255, 0.48235294, 0.44666667])
STD = np.array([63.0, 62.1, 66.7]) / 255.0  # = np.array([0.24705882, 0.24352941, 0.26156863])


class Cutout:
    
    """Randomly mask out a patch from an image.
    Args:
        size (int): The size of the square patch.
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image
        Returns:
            Tensor: Image with a hole of dimension size x size cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)
        
        mask[y1: y2, x1: x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img



def build_cifar_loaders(
        batch_size,
        eval_batch_size,
        data_dir,
        name,
        validation=True,
        workers=8,
        augment=False,
        reshuffle=True,
    ):

    location = data_dir + name + "/"
    

    
    # define transforms
    normalize = transforms.Normalize(
        mean=MEAN,
        std=STD,
    )
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(16),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    
    if name == "cifar10":
        dataset_class = datasets.CIFAR10
        n_classes = 10
    elif name == "cifar100":
        dataset_class = datasets.CIFAR100
        n_classes = 100
    else:
        raise ValueError("Unknown dataset name.")
    
    n_inputs = 3

    # load the dataset
    train_dataset = dataset_class(
        root=location, train=True,
        download=False, transform=train_transform,
    )
    
    test_dataset = dataset_class(
        root=location, train=False,
        download=False, transform=valid_transform,
    )

    if validation:
        
        valid_dataset = dataset_class(
            root=location, train=True,
            download=False, transform=valid_transform,
        )
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        
        if reshuffle:
            np.random.shuffle(indices)
        
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=workers, pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=eval_batch_size, sampler=valid_sampler,
            num_workers=workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
        )
        valid_loader = None

    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    
    image_size = 32
    return loaders, n_inputs, image_size, n_classes, None


if __name__ == "__main__":
        train_loader, valid_loader, test_loader, n_inputs, n_classes = build_cifar_loaders(
            batch_size=128,
            eval_batch_size=128,
            validation=True,
            workers=8,
            augment=False,
            reshuffle=True,
        )
        
        print(len(train_loader.dataset))
        print(len(valid_loader.dataset))
        print(len(test_loader.dataset))
    
        print(n_inputs)
        print(n_classes)
    
        for i, (images, labels) in enumerate(train_loader):
            print(images.shape)
            print(labels.shape)
            break
    
        for i, (images, labels) in enumerate(valid_loader):
            print(images.shape)
            print(labels.shape)
            break
    
        for i, (images, labels) in enumerate(test_loader):
            print(images.shape)
            print(labels.shape)
            break
    
        print("Done")
    
        # print(train_loader.dataset[0][0].shape)
        # print(train_loader.dataset[0][1].shape)
        # print(train_loader.dataset[0][1])
    
        # print(valid_loader.dataset[0][0].shape)
        # print(valid_loader.dataset[0][1].shape)
        # print(valid_loader.dataset[0][1])
    
        # print(test_loader.dataset[0][0].shape)
        # print(test_loader.dataset[0][1].shape)
        # print(test_loader.dataset[0][1])
    
        # print(len(train_loader.dataset))
        # print(len(valid_loader.dataset))
        # print(len(test_loader.dataset))
    
        # print(n_inputs)
        # print(n_classes)
    
        # for i, (images, labels) in enumerate(train_loader):
        #     print(images.shape)
        #     print(labels.shape)
        #     break
    
        # for i, (images, labels) in enumerate(valid_loader):
        #     print(images.shape)
        #     print(labels.shape)
        #     break
    
        # for i, (images, labels) in enumerate(test_loader):
        #     print(images.shape)
        #     print(labels.shape)
        #     break
    
        # print("Done")
    
        # print(train_loader.dataset[0][0].shape)
        # print(train_loader.dataset[0][1].shape)
        # print(train_loader.dataset[0][1])
    
        # print(valid_loader.dataset[0][0].shape)
        # print(valid_loader.dataset[0][1].shape)
        # print(valid_loader.dataset
