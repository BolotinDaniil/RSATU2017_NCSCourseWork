import numpy as np

import numpy as np
import csv
import pandas as pd


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
        df = pd.read_csv(file_name, sep=';')
        # f = open(file_name, encoding='utf-8')
        # csv_iter = csv.reader(f, delimiter=';')
        # self.data_name = next(csv_iter)
        # n = len(self.data_name)
        # n - count colums
        self.data = [[], [], []]
        for ci in df.iterrows():
            self.data[2].append(float(ci[1][7]))
            self.data[1].append(int(ci[1][3]))
            self.data[0].append(int(ci[1][2]))
        n = len(self.data[1])
        self.data[0] = [i for i in range(n)]
        print(self.data[0])
        print(self.data[2])

    def _get_one_example(self, data, future_cost):
        mmin = min(data)
        mmax = max(data)
        if mmax == mmin:
            mmax += 1
        mmin = mmin - (mmax - mmin) * 0.1
        mmax = mmax + (mmax - mmin) * 0.05
        llen = float(mmax - mmin)
        ndata = [float(i - mmin) / llen for i in data]
        if (future_cost > data[-1]):
            y = 1
        else:
            y = 0
        return ndata, y

    def parse_for_visual_analys(self, len_group=24, len_order=1):
        '''
        :param len_group: size of the sliding window
        :param len_order: forecast horizon
        :return: matrix x with len_group columns, and array y
        '''
        X = []
        y = []
        for i in range(len_group, len(self.data[2]) - len_order + 1):
            ndata = self.data[2][i - len_group:i]
            XX, yy = self._get_one_example(ndata, self.data[2][i + len_order - 1])
            X.append(XX)
            y.append(yy)

        self.X = np.array(X)
        self.y = np.array(y)
        return self.X, self.y

    def _get_split_points(self, valid_share, test_share):
        '''
        :param valid_share:
        :param test_share:
        :return:(val_split, test_split) for raw data, (val_split, test_split) for clear data
        '''
        nr = len(self.data[2])
        raw = (int(nr * (1 - test_share - valid_share) ), int(nr * (1 - test_share) ))
        clean = (raw[0] - self.len_group - self.len_order + 1, raw[1] - self.len_group - self.len_order + 1)
        return raw, clean

    def get_raw_data(self):
        '''
        :param valid_share:
        :param test_share:
        :return: train, valid, test dataset
        '''
        return self.raw_train, self.raw_valid, self.raw_test


    def get_all_raw_data(self):
        return self.data[2]

    def get_all_dataset(self):
        return self.X, self.y

    def get_dataset(self):
        return self.train, self.valid, self.test


    def load_data(self, file_name, len_group, len_order, valid_share, test_share):
        '''
        Laad and precalc data
        :param file_name:
        :param len_group:
        :param len_order:
        :param valid_share:
        :param test_share:
        :return: None
        '''
        self.file_name = file_name
        self.len_group = len_group
        self.len_order = len_order

        self.load_raw_data(file_name)
        self.parse_for_visual_analys(len_group, len_order)

        points_raw, points_clear = self._get_split_points(valid_share, test_share)
        val_split, test_split = points_raw
        self.raw_train = self.data[2][:val_split]
        self.raw_valid = self.data[2][val_split:test_split]
        self.raw_test = self.data[2][test_split:]

        val_split, test_split = points_clear
        self.train = (self.X[:val_split], self.y[:val_split])
        self.valid = (self.X[val_split:test_split], self.y[val_split:test_split])
        self.test = (self.X[test_split:], self.y[test_split:])