import numpy as np
from config_parser import config
from mlp import MLP, DenseLayer, SoftmaxLayer, ReLU
from dataset import DataSrc
from utils import timeit
import theano

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
        self.amount_winners = config['GA']['amount_winners']

        layers = [DenseLayer(5, 64, ReLU), SoftmaxLayer(64, 2)]
        self.nn = MLP(layers, 32)

        datasrc = DataSrc()
        X, y, X_val, y_val = datasrc.load()
        self.X = theano.shared(X)
        self.y = theano.shared(y.astype('int32'))
        self.X_val = theano.shared(X_val)
        self.y_val = theano.shared(y_val.astype('int32'))


    def _generate_gen(self, rand=None):
        if rand is None:
            rand = np.random.rand()
        return rand * 2 * config['GA']['max_abs_init_weight'] - config['GA']['max_abs_init_weight']

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

        cur = 0
        res = []
        for i in range(1, len(config['MLP']['layers'])):
            w = np.array(genotype[cur: cur + config['MLP']['layers'][i] * config['MLP']['layers'][i-1]])
            cur += config['MLP']['layers'][i] * config['MLP']['layers'][i-1]
            b = np.array(genotype[cur: cur + config['MLP']['layers'][i]])
            cur += config['MLP']['layers'][i]
            res.append(w)
            res.append(b)
        return res

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

        self.nn.fit_SGD((self.X, self.y), 4, 32, 0.1, (self.X_val, self.y_val), (self.X_val, self.y_val),
                        verbose=config['MLP']['verbose'],
                        weights_biases=init_weights)
        r = self.nn.evaluate((self.X_val, self.y_val))
        return r


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

if __name__ == '__main__':
    ga = GA(population_size=config['GA']['population_size'],
            genotype_size=config['GA']['genotype_size'],
            amount_generations=config['GA']['amount_generations'],
            crossing_rate=config['GA']['crossing_rate'],
            mutation_rate=config['GA']['mutation_rate'])
    ga.evalution()