from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import optimizers
import numpy as np
from utils import timeit
import math
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from config_parser import config


class keras_MLP(object):
    def __init__(self, *k, **params):
        pass

    def build_mode(self, batch_size, weights_biases):
        model = Sequential()
        model.add(Dense(config['MLP']['layers'][1], input_dim=config['MLP']['layers'][0],
                        activation=config['MLP']['activation_fns'][0],
                        weights=(weights_biases[0], weights_biases[1])
                        ))
        for i in range(2, len(config['MLP']['layers'])):
            layer = Dense(config['MLP']['layers'][i],
                          activation=config['MLP']['activation_fns'][i - 1],
                          weights=(weights_biases[2 * i - 2], weights_biases[2 * i - 1])
                          )
            model.add(layer)
        optimizer = optimizers.SGD(lr=config['MLP']['learning_rate'])

        if config['MLP']['layers'][-1] == 1:
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def fit(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0, verbose=0, weights_biases=None):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data

        x = training_x.get_value()

        val_x = validation_x.get_value()

        if config['MLP']['layers'][-1] == 1:
            y = training_y.get_value()
            val_y = validation_y.get_value()
        else:
            val_y = []
            for i in validation_y.get_value():
                if i == 0:
                    val_y.append([1, 0])
                else:
                    val_y.append([0, 1])
            y = []
            for i in training_y.get_value():
                if i == 0:
                    y.append([1, 0])
                else:
                    y.append([0, 1])

        self.build_mode(mini_batch_size, weights_biases)

        hist = self.model.fit(x, y,
                              validation_data=(val_x, val_y),
                              nb_epoch=config['MLP']['nb_epoch'],
                              batch_size=mini_batch_size,
                              verbose=config['MLP']['verbose'],
                              shuffle=False
                              )
        return hist.history['acc'], hist.history['val_acc']

    def predict(self, X):
        pred = self.model.predict(X)
        res = []
        if config['MLP']['layers'][-1] == 1:
            res = (int(i+0.5) for i in pred)
        else:
            for i in pred:
                res.append(1 if i[0] > i[1] else 0)
        return res
