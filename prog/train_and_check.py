import os
import cProfile
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['THEANO_FLAGS'] = 'device=cpu, floatX=float32'
# os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran'
import theano
theano.config.openmp = True
# Create first network with Keras
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
import settings

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

indexes = []
scores = []

def sortByIndex(index):
    global scores
    return scores[index]

class GA:
    def __init__(self, population_size, genotype_size, amount_generations, crossing_rate, mutation_rate):
        self.pop_size = population_size
        self.genotype_size = genotype_size
        self.amount_generations = amount_generations
        self.c_r = crossing_rate
        self.m_r = mutation_rate
        self.amount_winners = settings.amount_winners
        self.nn = Teacher()
        self.nn.load()

    def _generate_gen(self, rand=None):
        if rand is None:
            rand = np.random.rand()
        return rand * 2 * settings.max_abs_init_weight - settings.max_abs_init_weight

    def build_inital_polulation(self):
        self.population = self._generate_gen(np.random.random((self.pop_size, self.genotype_size)))

    def conv_genotype(self, genotype):
        w1 = np.array(genotype[0:5*64])
        w1.shape = (5, 64)
        b1 = np.array(genotype[5*64:6*64])
        w2 = np.array(genotype[6*64: 6*64+64])
        w2.shape = (64, 1)
        b2 = np.array(genotype[7*64])
        b2.shape = (1)
        return (w1, b1, w2, b2)

    def mutate_genotype(self, genotype_index):
        gi = genotype_index
        for i in range(int(np.random.randint(0, self.genotype_size //2 * 3))):
            self.population[gi][np.random.randint(0, self.genotype_size)] = self._generate_gen()

    def crossing_genotypes(self, p1, p2):
        '''
        :param p1: parent1: genotype
        :param p2: parent2
        :return: new genotype
        '''
        # split_point = np.random.randint(0, self.genotype_size)
        # g = np.concatenate( (p1[:split_point], p2[split_point:]), axis=0)
        g = np.empty( (self.genotype_size), dtype='float32')
        for i in range(self.genotype_size):
            if np.random.choice((True, False)):
                g[i] = p1[i]
            else:
                g[i] = p2[i]
        return g
        pass

    def crossing(self):
        new_pop = np.empty( (int(self.pop_size * self.c_r), self.genotype_size), dtype='float32')
        new_pop[0:self.pop_size] = self.population[:]
        self.population = new_pop
        for i in range(int((self.c_r-1) * self.pop_size)):
            p1 = np.random.randint(0, self.pop_size)
            p2 = np.random.randint(0, self.pop_size)
            self.population[i+self.pop_size] = self.crossing_genotypes(self.population[p1], self.population[p2])
            if np.random.rand() < self.m_r:
                self.mutate_genotype(i+self.pop_size)


    def fit_genotype(self, genotype):
        init_weights = self.conv_genotype(genotype)
        self.nn.build_model(init_weights)
        _, r = self.nn.train(settings.nb_epoch, settings.batch_size)
        return r[-1]


    @timeit
    def selection(self):
        global indexes, scores
        indexes = []
        scores = []
        for i in range(self.population.shape[0]):
            indexes.append(i)
            scores.append(self.fit_genotype(self.population[i]))
        indexes.sort(key=sortByIndex, reverse=True)
        new_population = np.empty((self.pop_size, self.genotype_size), dtype='float32')
        new_scores = np.empty( (self.pop_size), dtype='float32')
        for i in range(self.amount_winners):
            new_population[i] = self.population[indexes[i]]
            new_scores[i] = scores[indexes[i]]
        for i in range(self.amount_winners, self.pop_size):
            index = [indexes[np.random.randint(self.amount_winners, self.population.shape[0])]]
            new_population[i] = self.population[index]

        self.population = new_population

        # heuristic
        if np.var(new_scores)**0.5 < 0.03:
            self.amount_winners = 14
        elif np.var(new_scores)**0.5 > 0.2:
            self.amount_winners = 20
        else:
            self.amount_winners = 20

        print('generation: {}, mean: {}, deviation: {}, max: {}'
              .format(self.number_generation, np.mean(new_scores), np.var(new_scores)**0.5, new_scores[0]))


    def evalution(self):
        self.build_inital_polulation()
        for i in range(self.amount_generations):
            self.number_generation = i + 1
            self.crossing()
            self.selection()

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


    def build_model(self, weights):
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        # load dataset

        # create model
        #convolution model
        self.model = Sequential()

        self.model.add(Dense(64, input_dim=self.len_group, activation='relu',
                             weights=(weights[0], weights[1])))
        # self.model.add(Dense(32, init='he_normal', activation='relu'))
        # self.model.add(Dense(8, init='he_normal', activation='relu'))
        # self.model.add(Dense(5, init='he_normal', activation='relu'))
        # self.model.add(Dense(32, init='he_normal', activation='relu'))
        self.model.add(Dense(1, activation='sigmoid',
                             weights=(weights[2], weights[3])))
        # Compile model
        optimizer = optimizers.SGD(lr=1e-3)
        # optimizer = optimizers.RMSprop()
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    # @timeit
    def train(self, nb_epoch, batch_size):
        res = self.model.fit(self.X_train, self.Y_train,
                             validation_data=(self.X_test, self.Y_test),
                             nb_epoch=nb_epoch, batch_size=batch_size, verbose=settings.verbose)
        res_train = res.history['acc']
        res_test = res.history['val_acc']
        return res_train, res_test




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
    # t = Teacher()
    # t.load()
    # t.build_model()
    # res_train, res_test = t.train(nb_epoch=settings.nb_epoch, batch_size=settings.batch_size)
    # plt.plot(res_train, color='blue')
    # plt.plot(res_test, color='red')
    # plt.grid(True)
    # plt.show()
    # t.standart_evaluate(t.X_test, t.Y_test, show=True)
    # t.traiding_evaluate(t.X_test, t.Y_test)
    ga = GA(population_size=settings.population_size,
            genotype_size=settings.genotype_size,
            amount_generations=settings.amount_generations,
            crossing_rate=settings.crossing_rate,
            mutation_rate=settings.mutation_rate)
    ga.evalution()
