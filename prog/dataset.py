import numpy as np

import numpy as np
import csv


class DataSrc():
    def __init__(self):
        self.file_name = ""

    def load_raw_data(self, file_name):
        """
                load and pre calc data
                :param file_name:
                :return: None
        """
        print()
        print("file = ", file_name)
        f = open(file_name)
        # f = open(file_name, encoding='utf-8')
        csv_iter = csv.reader(f, delimiter=';')
        self.data_name = next(csv_iter)
        n = len(self.data_name)
        # n - count colums
        self.data = [[], [], []]
        for ci in csv_iter:
            self.data[2].append(float(ci[7]))
            self.data[1].append(int(ci[3]))
            self.data[0].append(int(ci[2]))
        n = len(self.data[1])
        self.data[0] = [i for i in range(n)]
        print(self.data[0])
        print(self.data[2])

    def get_raw_data(self):
        return self.data

    def load_data(self):
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