from os import listdir
from os.path import isfile, join
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cfg
import cv2


def get_data(path, target):
    X = []
    y = []
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for img in files:
        X.append(plt.imread(path + img))
        y.append(target)
    return X, y


def prepare_data(X_train, y_train, X_test, y_test):
    size = (cfg.IMG_SIZE, cfg.IMG_SIZE)
    for i, image in enumerate(X_train):
        X_train[i] = tf.image.resize(image, size).numpy()

    for i, image in enumerate(X_test):
        X_test[i] = tf.image.resize(image, size).numpy()
    y_trainOH = tf.one_hot(y_train, cfg.NUM_CLASSES)
    y_testOH = tf.one_hot(y_test, cfg.NUM_CLASSES)
    X_train = np.asarray([X_train[i].astype('float32') for i in range(len(X_train))])
    X_test = np.asarray([X_test[i].astype('float32') for i in range(len(X_test))])
    return X_train, y_trainOH, X_test, y_testOH


def get_dataset():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    i = 0

    print("Loading data...")

    for folder in ("TRAIN", "TEST"):
        print("Loading " + folder + " data...")
        for lesion_type in ("NEV", "MEL"):
            print("Loading " + lesion_type + " data...")
            current_path = cfg.PATH + folder + "/" + lesion_type + "/"
            files = [f for f in listdir(current_path) if isfile(join(current_path, f))]
            for img in files:
                if folder == "TRAIN":
                    X_train.append(plt.imread(current_path + img))
                    y_train.append(1 if lesion_type == "MEL" else 0)
                else:
                    X_test.append(plt.imread(current_path + img))
                    y_test.append(1 if lesion_type == "MEL" else 0)
                i += 1

                sys.stdout.write('\r')
                sys.stdout.write(str(round(i / 6126. * 100, 2)) + "%")
                if i > 200:
                    break
            print("\nDone!")
    return prepare_data(X_train, y_train, X_test, y_test)


def get_testset():
    X_test = []
    y_test = []
    for lesion_type in ("NEV", "MEL"):
        current_path = cfg.PATH + 'TEST/' + lesion_type + "/"
        files = [f for f in listdir(current_path) if isfile(join(current_path, f))]
        for img in files:
            X_test.append(tf.image.resize(plt.imread(current_path + img), (cfg.IMG_SIZE, cfg.IMG_SIZE)).numpy())
            y_test.append(1 if lesion_type == "MEL" else 0)
    return X_test, keras.utils.to_categorical(y_test, num_classes=cfg.NUM_CLASSES)


def mel_augmentation(X, y):
    addedMel = []
    for i, img in enumerate(X):
        if y[i] == 1:
            addedMel.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
            addedMel.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
            addedMel.append(cv2.flip(img, 0))
            addedMel.append(cv2.flip(img, 1))
    return addedMel


# X_train, y_train, X_test, y_test = get_dataset(PATH)


# Data generator
# In case of lack of space

def get_dict_dataset(path):
    partition = {}
    labels = {}
    partition['TRAIN'] = []
    partition['TEST'] = []
    for folder in ("TRAIN", "TEST"):
        for lesion_type in ("NEV", "MEL"):
            current_path = path + folder + "/" + lesion_type + "/"
            files = [f for f in listdir(current_path) if isfile(join(current_path, f))]
            partition[folder] += [current_path + f for f in files]
            for f in files:
                labels[current_path + f] = 0 if lesion_type == 'NEV' else 1
    return partition, labels


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=3,
                 n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = tf.image.resize(plt.imread(ID), self.dim).numpy()
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def get_generators(PATH):
    params = {
        'dim': (cfg.IMG_SIZE, cfg.IMG_SIZE),
        'batch_size': cfg.batch_size,
        'n_classes': 2,
        'n_channels': 3,
        'shuffle': True
    }

    partition, labels = get_dict_dataset(PATH)
    training_generator = DataGenerator(partition['TRAIN'], labels, **params)
    test_generator = DataGenerator(partition['TEST'], labels, **params)

    return training_generator, test_generator

# train_g, test_g = get_generators(PATH)
