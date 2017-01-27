import numpy as np
from config_parser import config
from mlp import MLP, DenseLayer, SoftmaxLayer, ReLU
from dataset import DataSrc
from utils import timeit
import theano

from mlp import MLP, DenseLayer, SoftmaxLayer, ReLU
from genetics import GA
from config_parser import config

class GA_MLP:
    def __init__(self):
        pass

    def fit(self, main_window):
        print('run_fit')
        layers = [DenseLayer(5, 64, ReLU), SoftmaxLayer(64, 2)]
        mlp = MLP(layers, config['MLP']['batch_size'])
        X, y, X_val, y_val = main_window.data_src.load_data()
        ga = GA(population_size=config['GA']['population_size'],
                genotype_size=config['GA']['genotype_size'],
                amount_generations=config['GA']['amount_generations'],
                crossing_rate=config['GA']['crossing_rate'],
                mutation_rate=config['GA']['mutation_rate'],
                mlp = mlp,
                train_data = (X, y),
                val_data = (X_val, y_val))
        ga.build_inital_polulation()
        for i in range(config['GA']['amount_generations']):
            scores, hist = ga.step_evolution()
            main_window.label_best_acc.setText(str(scores[0]))
            # break

def fit_model(main_window):
    m = GA_MLP()
    m.fit(main_window)
