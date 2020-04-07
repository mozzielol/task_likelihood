import tensorflow as tf
import numpy as np
from copy import deepcopy
from keras.datasets import mnist
from keras.utils import np_utils
import os

def construct_split_mnist(task_labels,  split='train', flatten=True,multihead=False):

    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = np.reshape(np.array(X_train),newshape=[-1,28,28,1])
        X_test = np.reshape(np.array(X_test),newshape=[-1,28,28,1])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    return split_dataset_by_labels(X, y, task_labels, nb_classes, multihead)


def split_dataset_by_labels(X, y, task_labels, nb_classes=None, multihead=False):

    if nb_classes is None:
        nb_classes = len(np.unique(y))

    datasets = []
    for labels in task_labels:
        idx = np.in1d(y, labels)
        if multihead:
            label_map = np.arange(nb_classes)
            label_map[labels] = np.arange(len(labels))
            data = X[idx], np_utils.to_categorical(label_map[y[idx]], len(labels))
        else:
            data = X[idx], np_utils.to_categorical(y[idx], int(nb_classes))
        datasets.append(data)
    return datasets


def construct_permute_mnist(num_tasks=2,  split='train', permute_all=False, subsample=1):
    """Create permuted MNIST tasks.
        Args:
                num_tasks: Number of tasks
                split: whether to use train or testing data
                permute_all: When set true also the first task is permuted otherwise it's standard MNIST
                subsample: subsample by so much
        Returns:
            List of (X, y) tuples representing each dataset
    """
    # Load MNIST data and normalize
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_train, y_train = X_train[::subsample], y_train[::subsample]
    X_test, y_test = X_test[::subsample], y_test[::subsample]

    permutations = []
    # Generate random permutations
    for i in range(num_tasks):
        idx = np.arange(X_train.shape[1],dtype=int)
        if permute_all or i>0:
            np.random.shuffle(idx)
        permutations.append(idx)

    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    X,y = X_train,y_train
    train_set = []
    for perm in permutations:
        data = X[:,perm], np_utils.to_categorical(y, nb_classes)
        train_set.append(data)

    X,y = X_test,y_test
    test_set = []
    for perm in permutations:
        data = X[:,perm], np_utils.to_categorical(y, nb_classes)
        test_set.append(data)
    return train_set,test_set


def construct_split_ucr(task_labels, dataset_name = 'Two_Patterns', split='train',dataset_folder='./data/'):
    X_train, y_train, X_test, y_test = load_dataset(dataset_name,dataset_folder)
    nb_classes = np.max(y_train)+1
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    return split_dataset_by_labels(X, y, task_labels, nb_classes)

def load_dataset(dataset_name, dataset_folder):
    dataset_path = os.path.join(dataset_folder, dataset_name)
    train_file_path = os.path.join(dataset_path, '{}_TRAIN'.format(dataset_name))
    test_file_path = os.path.join(dataset_path, '{}_TEST'.format(dataset_name))

    # training data
    train_raw_arr = np.genfromtxt(train_file_path, delimiter=',')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = np.genfromtxt(test_file_path, delimiter=',')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    return train_data, train_labels, test_data, test_labels

