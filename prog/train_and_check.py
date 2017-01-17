import os
import cProfile
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['THEANO_FLAGS'] = 'device=cpu, floatX=float64'
# os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran'
import theano
theano.config.openmp = True
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np

def get_class_labes(res):
    m = 0
    r = 0
    for i in range(len(res)):
        if res[i] > m:
            m = res[i]
            r = i
    return r

def my_init(shape, name=None):
    print("shape = ", shape)
    value = np.random.random(shape)
    return K.variable(value, name=name)

class Teacher:
    model = None
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


    def make_model(self):
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        # load dataset

        # create model
        #convolution model
        self.model = Sequential()
        # self.model.add(Convolution1D(64, 3, border_mode='same', input_shape = (1,  self.len_group), activation='relu'))
        # self.model.add(Convolution1D(64, 5, border_mode='same', activation='relu'))
        # self.model.add(Convolution1D(64, 7, border_mode='same', activation='relu'))
        # self.model.add(Convolution1D(32, 5, border_mode='same', activation='relu'))
        # self.model.add(Flatten())
        # self.model.add(Dense(32, init='he_normal', activation='relu'))
        # self.model.add(Dense(16, init='he_normal', activation='relu'))
        # self.model.add(Dense(8, init='he_normal', activation='relu'))
        # self.model.add(Dense(3, init='uniform', activation='sigmoid'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # self.X_train = self.X_train.reshape(-1, 1, self.len_group)
        # self.X_test = self.X_test.reshape(-1, 1, self.len_group)
        # fully connected model
        self.model.add(Dense(64, input_dim=self.len_group, init=my_init, activation='relu'))
        # self.model.add(Dense(32, init='he_normal', activation='relu'))
        # self.model.add(Dense(8, init='he_normal', activation='relu'))
        # self.model.add(Dense(5, init='he_normal', activation='relu'))
        # self.model.add(Dense(32, init='he_normal', activation='relu'))
        self.model.add(Dense(1, init=my_init, activation='sigmoid'))
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        # start_time = time.time()
        # self.model.fit(self.X_train, self.Y_train, nb_epoch=50, batch_size=32)
        # evaluate the model
        # print('\nTrain take %s minutes' % (int(time.time() - start_time) // 60))

        # print('\n\nIn train data')
        # self.standart_evaluate(self.X_train, self.Y_train)
        # print('\n\nIn test data')
        # self.standart_evaluate(self.X_test, self.Y_test)

    def train_and_fix_learning(self, nb_epoch, batch_size):
        res_train = []
        res_test = []
        for i in range(nb_epoch):
            print("\nepoch = %s" % i)
            self.model.fit(self.X_train, self.Y_train, nb_epoch=1, batch_size=batch_size)
            res_train.append(self.standart_evaluate(self.X_train, self.Y_train, show=False))
            res_test.append(self.standart_evaluate(self.X_test, self.Y_test, show=False))

        plt.plot(res_train, color = 'blue')
        plt.plot(res_test, color = 'red')
        plt.grid(True)
        plt.show()


    def standart_evaluate(self, X, Y, show=True):
        scores = self.model.evaluate(X, Y)
        for i in range(len(scores)):
            if show: print("%s: %.2f%%" % (self.model.metrics_names[i], scores[i] * 100))
        return scores[1] # acc

    def traiding_evaluate(self, X, Y):
        res = self.model.predict(X, batch_size = 10, verbose = 0)
        # print res
        # for i in range(len(res)):
        #     res[i][0] = int(res[i][0] + 0.5)
        #     res[i][1] = int(res[i][1] + 0.5)
        #     res[i][2] = int(res[i][2] + 0.5)

        # print res
        score = 0
        cnt_orders = 0
        for i in range(len(res)):
            if get_class_labes(res[i]) == 1:
                cnt_orders += 1
                if Y[i] == 1:
                    score += 2
                elif Y[i] == 0:
                    score -= 1
                else :
                    score -= 2
            if get_class_labes(res[i]) == 2:
                cnt_orders += 1
                if Y[i] == 2:
                    score += 2
                elif Y[i] == 0:
                    score -= 1
                else:
                    score -= 2
        print('count orders = %s' % cnt_orders)
        print('Profit = %s' % score)


if __name__ == "__main__":
    t = Teacher()
    t.load()
    t.make_model()
    t.train_and_fix_learning(nb_epoch=150, batch_size=32)
    t.standart_evaluate(t.X_test, t.Y_test, show=True)
    # t.traiding_evaluate(t.X_test, t.Y_test)
