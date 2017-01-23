import matplotlib.pyplot as plt
import numpy as np
import csv

import pandas as pd
import statsmodels.api as sm
import patsy as pt

class DataParser:

    data = []

    def __init__(self, file_name = None):
        if file_name != None:
            self.load(file_name)

    def get_inital_moment2(self, seq, k):
        r = 0
        for xi in seq:
            r += xi ** k
        r /= len(seq)
        return r

    def get_central_moment2(self, seq, k):
        mx = self.get_inital_moment2(seq, 1)
        return self.get_inital_moment2([xi - mx for xi in seq], k)

    def load(self, file_name):
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
        self.data = [[],[],[]]
        for ci in csv_iter:
            self.data[2].append(float(ci[7]))
            self.data[1].append(int(ci[3]))
            self.data[0].append(int(ci[2]))
        n = len(self.data[1])
        self.data[0] = [i for i in range(n)]
        print(self.data[0])
        print(self.data[2])

    def show_data(self):
        x = np.arange(0,len(self.data[2]),1)
        plt.plot(x, self.data[2])
        plt.grid(True)
        plt.show()
        pass

    def parse_and_save_data(self, len_group = 24, len_test = 240):
        ndata = []
        for i in range(1,len(self.data[2])):
            ndata.append(self.data[2][i] - self.data[2][i-1])
        self.data[0] = self.data[0][1:]
        self.data[1] = self.data[1][1:]
        self.data[2] = ndata
        self.show_data()

        f = open("data/train.csv", 'wb')
        writer = csv.writer(f, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len_group, len(self.data[2]) - len_test):
            if self.data[2][i] > 0:
                out = [1]#[1, 0]
            else :
                out = [0]#[0,1]
            writer.writerow(self.data[2][i-len_group:i] + out)
        f.close()

        f = open("data/test.csv", 'wb')
        writer = csv.writer(f, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(self.data[2]) - len_test, len(self.data[2])):
            if self.data[2][i] > 0:
                out = [1]#[1, 0]
            else :
                out = [0]#[0,1]
            writer.writerow(self.data[2][i - len_group:i] + out)
        f.close()

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
            ndata.append(1)
        else:
            ndata.append(0)
        return ndata

    def parse_for_visial_analys(self, len_group = 24,len_order = 1, len_test = 30):
        #return data in the form
        #that we can  see in traiding platform
        f = open("data/train.csv", 'wb')
        writer = csv.writer(f, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len_group, len(self.data[2]) - len_test - len_order + 1):
            ndata = self.data[2][i-len_group:i]
            r = self._get_one_example(ndata, self.data[2][i+len_order-1])
            writer.writerow(r)
        f.close()

        f = open("data/test.csv", 'wb')
        writer = csv.writer(f, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(self.data[2]) - len_test - len_order + 1, len(self.data[2]) - len_order + 1):
            ndata = self.data[2][i - len_group:i]
            r = self._get_one_example(ndata, self.data[2][i + len_order - 1])
            writer.writerow(r)
        f.close()


    def get_data(self, count_items = None):
        if count_items == 0:
            return self.data
        else: return [self.data[0][:count_items], self.data[1][:count_items]]

if __name__ == "__main__":
    d = DataParser()
    # d.load("GOLD_051101_161101_1DAY.csv")
    # d.parse_and_save_data(len_group = 20, min_border = 4,  len_test = 30 * 5)
    # d.load("BRENT_141101_161101_1DAY.csv")

    d.load("data/BRENT_001101_161101_1DAY.csv")
    d.parse_for_visial_analys(len_group = 5, len_order = 1, len_test = 30 * 5)
    # d.parse_and_save_data(len_group = 8,  len_test = 30 * 5)