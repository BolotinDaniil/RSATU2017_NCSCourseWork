
from dataset import DataSrc

#### Libraries
# Standard library
# import cPickle
import gzip

# Third-party libraries
import os
import numpy as np
import theano
import copy
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = False

# init theano
# theano.config.exception_verbosity = 'high'
# theano.config.mode = 'DebugMode'
theano.config.mode = 'FAST_RUN'
if GPU:
    print("Trying to run under a GPU")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running on CPU. ")

#### Main class used to construct and train networks
class MLP(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def set_weights(self, weights_biases):
        for i in range(1, len(self.layers)):
            self.layers[i-1].w = theano.shared(weights_biases[2*i - 2], name='w', borrow=True)
            self.layers[i-1].b = theano.shared(weights_biases[2*i - 1], name='b', borrow=True)
            self.layers[i-1].params = [self.layers[i-1].w, self.layers[i-1].b]

        self.params = [param for layer in self.layers for param in layer.params]
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def fit_SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0, verbose=0, weights_biases=None):
        '''
        training via backpropagation

        :param training_data:
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param validation_data:
        :param test_data:
        :param lmbda:
        :param weights_biases: array inital weights and biases
        :return:
        '''
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)//mini_batch_size
        num_validation_batches = size(validation_data)//mini_batch_size
        num_test_batches = size(test_data)//mini_batch_size

        self.mini_batch_size = mini_batch_size

        # apply weights
        if not weights_biases is None:
            self.set_weights(weights_biases)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        val_acc = []
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    if verbose > 0: print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    val_acc.append(validation_accuracy)
                    if verbose > 0: print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        if verbose > 1: print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            if verbose > 0: print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        if verbose > 0: print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        if verbose > 0: print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))
        return val_acc

    def evaluate(self, data):
        X, y = data
        i = T.lscalar()  # mini-batch index
        ds = size(data)
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    X[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                    y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            }
          )
        acc = np.mean([test_mb_accuracy(i) for i in range(ds//self.mini_batch_size)])
        return acc

    def predict(self, X):
        '''
        :param X: numpy array
        :return: predicted values
        '''
        i = T.lscalar()  # mini-batch index
        ds = X.shape[0]
        # _X = np.copy(X)
        _X = copy.deepcopy(X)
        # print(_X)
        nb_empty = 0
        if ds % self.mini_batch_size > 0:
            nb_empty = self.mini_batch_size - (ds % self.mini_batch_size)
            for j in range(nb_empty):
                _X = np.concatenate((_X, [_X[-1]]))

        _th_X = theano.shared(_X)
        # th_X = theano.shared(X)
        predict_mb = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                    _th_X[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        out = []
        for j in range((ds + nb_empty) // self.mini_batch_size):
	        out = np.concatenate((out, predict_mb(j)))
        # remove unnecessary
        out = np.delete(out, np.s_[ds:ds+nb_empty], 0)
        return out

#### Define layer types

class DenseLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
             np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)

        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                    dtype=theano.config.floatX),
            name='b', borrow=True)

        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."

        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]
    # return data[0].shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)


def test_mlp():
    datasrc = DataSrc()
    X, y, X_val, y_val = datasrc.load()
    X = theano.shared(X)
    y = theano.shared(y.astype('int32'))
    X_val = theano.shared(X_val)
    y_val = theano.shared(y_val.astype('int32'))
    print(y_val)
    layers = [DenseLayer(5, 64, ReLU), SoftmaxLayer(64, 2)]
    m = MLP(layers, 32)
    m.fit_SGD((X, y), 40, 32, 0.1, (X_val, y_val), (X_val, y_val))
    print(m.evaluate((X_val, y_val)))

if __name__ == '__main__':
    test_mlp()