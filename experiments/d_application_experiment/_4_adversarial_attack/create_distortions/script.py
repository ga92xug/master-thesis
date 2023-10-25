import collections
from distortions_functions import *

random_seed = 42
# Split the data into train, val, and test arrays.
train_images, train_labels, val_images, val_labels, test_images, test_labels = split_function(images, labels, random_seed=random_seed, reduction_factor=reduction_factor)






def main():
    d = get_distortions_dict()
    get_ISIC_2019()

    test_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False)
    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])


    for method_name in d.keys():
        print('Creating images for the corruption', method_name)
        cifar_c, labels = [], []

        for severity in range(1,6):
            corruption = lambda clean_img: d[method_name](clean_img, severity)

            for img, label in zip(test_data.data, test_data.targets):
                labels.append(label)
                cifar_c.append(np.uint8(corruption(convert_img(img))))

        np.save('/share/data/vision-greg2/users/dan/datasets/CIFAR-10-C/' + d[method_name].__name__ + '.npy',
                np.array(cifar_c).astype(np.uint8))

        np.save('/share/data/vision-greg2/users/dan/datasets/CIFAR-10-C/labels.npy',
                np.array(labels).astype(np.uint8))