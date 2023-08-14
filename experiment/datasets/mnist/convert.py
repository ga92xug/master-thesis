
################################################################################
# MNIST_ROT
################################################################################

import numpy as np

location = "../../Data/frischs/datasets/mnist_rot/"
f = open(location + "mnist_all_rotation_normalized_float_test.amat", "r")

test = []

for line in f:
    test.append([float(x) for x in line.split()])

test = np.array(test)


f = open(location + "mnist_all_rotation_normalized_float_train_valid.amat", "r")

trainval = []

for line in f:
    trainval.append([float(x) for x in line.split()])

trainval = np.array(trainval)

train = trainval[:10000, :].copy()
valid = trainval[10000:, :].copy()

np.savez(location + "mnist_rot_trainval", images=trainval[:, :-1].reshape(-1, 28, 28), labels=trainval[:, -1])
np.savez(location + "mnist_rot_test", images=test[:, :-1].reshape(-1, 28, 28), labels=test[:, -1])
np.savez(location + "mnist_rot_train", images=train[:, :-1].reshape(-1, 28, 28), labels=train[:, -1])
np.savez(location + "mnist_rot_valid", images=valid[:, :-1].reshape(-1, 28, 28), labels=valid[:, -1])

del train
del valid
del test

np.random.shuffle(trainval)

train = trainval[:10000, :].copy()
valid = trainval[10000:, :].copy()

np.savez(location + "mnist_rot_train_shuffled", images=train[:, :-1].reshape(-1, 28, 28), labels=train[:, -1])
np.savez(location + "mnist_rot_valid_shuffled", images=valid[:, :-1].reshape(-1, 28, 28), labels=valid[:, -1])


################################################################################
# FLIPROT
################################################################################


np.random.seed(42)

def preprocess(dataset, flip_all=False):
    
    images = dataset[:, :-1].reshape(-1, 28, 28)
    labels = dataset[:, -1]
    
    if flip_all:
        # augment the dataset with a flipped copy of each datapoint    
        flipped_images = images[:, :, ::-1]
        
        images = np.concatenate([images, flipped_images])
        labels = np.concatenate([labels, labels])
    else:
        # for each datapoint, we choose randomly whether to flip it or not 
        idxs = np.random.binomial(1, 0.5, dataset.shape[0])
        
        images[idxs, ...] = images[idxs, :, ::-1]
    
    return {"images": images, "labels": labels}

location = "../../Data/frischs/datasets/mnist_rot/"
f = open(location + "mnist_all_rotation_normalized_float_test.amat", "r")

test = []

for line in f:
    test.append([float(x) for x in line.split()])

test = np.array(test)
np.savez(location + "mnist_fliprot_test", **preprocess(test, flip_all=True))
del test

f = open(location + "/mnist_all_rotation_normalized_float_train_valid.amat", "r")

trainval = []

for line in f:
    trainval.append([float(x) for x in line.split()])

npoints = len(trainval)

trainval = np.array(trainval)

trainval = preprocess(trainval)

np.savez(location + "mnist_fliprot_trainval", **trainval)
np.savez(location + "mnist_fliprot_train", images=trainval["images"][:10000, ...], labels=trainval["labels"][:10000, ...])
np.savez(location + "mnist_fliprot_valid", images=trainval["images"][10000:, ...], labels=trainval["labels"][10000:, ...])

idxs = np.arange(npoints)
np.random.shuffle(idxs)
trainval["images"] = trainval["images"][idxs, ...]
trainval["labels"] = trainval["labels"][idxs]

np.savez(location + "mnist_fliprot_train_shuffled", images=trainval["images"][:10000, ...], labels=trainval["labels"][:10000, ...])
np.savez(location + "mnist_fliprot_valid_shuffled", images=trainval["images"][10000:, ...], labels=trainval["labels"][10000:, ...])


