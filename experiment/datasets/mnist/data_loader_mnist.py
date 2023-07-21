from email.mime import image
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

from . import own_transforms


class MNIST_Dataset(data.Dataset):
    """ rotated MNIST dataset """
    
    def __init__(self, data_dir, name, mode, transform=None, target_transform=None, reshuffle_seed=None):
        """
        :type  mode: string from ['train', 'valid', 'test']
        :param mode: determines which subset of the dataset is loaded and whether augmentation is used
        :type  transform: callable
        :param transform: transformation applied to PIL images, returning transformed version
        :type  target_transform: callable
        :param target_transform: transformation applied to labels
        :type  reshuffle_seed: int
        :param reshuffle_seed: seed to use to reshuffle train or valid sets. If None (default), they are not reshuffled
        """
        assert mode in ['train', 'valid', 'trainval', 'test']
        assert reshuffle_seed is None or (mode != "test" and mode != 'trainval')
        if name == "mnist12k":
            name = "mnist"
        
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        # load the numpy arrays
        if mode in ["train", "valid", "trainval"]:
            
            filename = f'/{name}_trainval.npz'
            
            data = np.load(data_dir + filename)

            num_train = len(data["labels"])
            indices = np.arange(0, num_train)

            if reshuffle_seed is not None:
                rng = np.random.RandomState(reshuffle_seed)
                rng.shuffle(indices)

            split = int(np.floor(num_train * 5/6))
            
            if mode == "train":
                data = {
                    "images": data["images"][indices[:split], :],
                    "labels": data["labels"][indices[:split]]
                }
            elif mode == "valid":
                data = {
                    "images": data["images"][indices[split:], :],
                    "labels": data["labels"][indices[split:]]
                }
            
        else:
            filename = f'/{name}_test.npz'
            data = np.load(data_dir + filename)

        self.images = data['images'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        """
        :type  index: int
        :param index: index of data
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, label = self.images[index], self.labels[index]
        # convert to PIL Image
        image = Image.fromarray(image)
        # transform images and labels
        if self.transform is not None:
            self.transform.update_randomization()
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
    
    def __len__(self):
        return len(self.labels)


def build_mnist_rot_loader(
        batch_size,
        data_dir,
        name,
        mode, 
        num_workers=8,
        drop_last=False, 
        rot_interpol_augmentation=False, 
        interpolation=0, 
        reshuffle_seed=None, 
        coords=False,
    ):
    """  """
    if mode is None:
        return None, None, None

    assert mode in ['train', 'valid', 'trainval', 'test']
    assert reshuffle_seed is None or (mode != "test" and mode != 'trainval')

    transform, shuffle = get_transform(name=name, mode=mode, rot_interpol_augmentation=rot_interpol_augmentation, interpolation=interpolation, coords=coords)

    location = data_dir + name
    
    dataset = MNIST_Dataset(data_dir=location, name=name, mode=mode, 
                            transform=transform, reshuffle_seed=reshuffle_seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    n_inputs = 1
    n_outputs = 10
    
    if coords:
        n_inputs += 2

    return loader, n_inputs, n_outputs


def get_transform(name, mode, rot_interpol_augmentation, interpolation, coords):
    rng = np.random.RandomState(42)

    if name == "mnist_rot":
        class_transform = own_transforms.Rotate
    elif name == "mnist_fliprot":
        class_transform = own_transforms.FlipRotate
    
    transform = []
    if mode in ['valid', 'test']:
        shuffle = False
        if rot_interpol_augmentation:
            transform = [
                class_transform(rng=None, interpolation=interpolation),  # only resamples image
                own_transforms.GrayToTensor()
            ]
        else:
            transform = [own_transforms.GrayToTensor()]
    elif mode in ['train', 'trainval']:
        shuffle = True
        if rot_interpol_augmentation:
            transform = [
                class_transform(rng=rng, interpolation=interpolation),
                own_transforms.GrayToTensor()
            ]
            
        else:
            transform = [
                own_transforms.GrayToTensor()
            ]
    else:
        raise ValueError('unknown mode for building mnist_rot loader')
    
    if coords:
        transform += [own_transforms.CoordinateField((28, 28))]
    
    return own_transforms.Compose(transform), shuffle


def build_mnist_loaders(
        batch_size,
        eval_batch_size,
        data_dir,
        name,
        reshuffle,
        validation,
        augment,
        workers=8, 
        interpolation=0, 
        coords=False,
        drop_last_train=False,
):
        if reshuffle:
            seed = np.random.randint(0, 100000)
        else:
            seed = None

        if validation:
            modes = ["train", "valid"]
        else:
            modes = ["trainval", None]
            
        train_loader, _, _ = build_mnist_rot_loader(
            batch_size=batch_size,
            data_dir=data_dir,
            name=name,
            mode=modes[0],
            num_workers=workers,
            rot_interpol_augmentation=augment,
            interpolation=interpolation,
            reshuffle_seed=seed,
            coords=coords,
            drop_last=drop_last_train,
        )
        valid_loader, _, _ = build_mnist_rot_loader(
            batch_size=eval_batch_size,
            data_dir=data_dir,
            name=name,
            mode=modes[1],
            num_workers=workers,
            rot_interpol_augmentation=False,
            interpolation=interpolation,
            coords=coords,
        )
        test_loader, n_inputs, n_outputs = build_mnist_rot_loader(
            batch_size=eval_batch_size,
            data_dir=data_dir,
            name=name,
            mode="test",
            num_workers=workers,
            rot_interpol_augmentation=False,
            interpolation=interpolation,
            coords=coords,
        )

        loaders = {
            "train": train_loader,
            "valid": valid_loader,
            "test": test_loader
        }
        image_size = 28
        return  loaders, n_inputs, image_size, n_outputs, None