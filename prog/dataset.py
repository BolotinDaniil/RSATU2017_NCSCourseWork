import numpy as np


class DataSrc():
    def __init__(self):
        pass

    def load(self):
        dataset_train = np.loadtxt("data/train.csv", delimiter=";")
        # split into input (X) and output (Y) variables
        self.len_group = len(dataset_train[0]) - 1
        self.X_train = dataset_train[:, 0:self.len_group]
        self.Y_train = dataset_train[:, self.len_group]

        dataset_test = np.loadtxt("data/test.csv", delimiter=";")
        # split into input (X) and output (Y) variables
        self.X_test = dataset_test[:, 0:self.len_group]
        self.Y_test = dataset_test[:, self.len_group]

        return self.X_train, self.Y_train, self.X_test, self.Y_test