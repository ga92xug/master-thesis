

import numpy as np

location = "../Data/frischs/mnist_rot/"
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

