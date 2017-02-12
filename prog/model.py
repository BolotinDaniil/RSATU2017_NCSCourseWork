import numpy as np
from config_parser import config
from mlp import MLP, DenseLayer, SoftmaxLayer, ReLU
from dataset import DataSrc
from utils import timeit
import theano
import threading

from mlp import MLP, DenseLayer, SoftmaxLayer, ReLU
from genetics import GA
from config_parser import config

class GA_MLP:
    def __init__(self, main_window):
        self.stop_event = threading.Event()
        layers = [DenseLayer(5, 64, ReLU), SoftmaxLayer(64, 2)]
        mlp = MLP(layers, config['MLP']['batch_size'])
        X, y, X_val, y_val = main_window.data_src.load_data()
        ga = GA(population_size=config['GA']['population_size'],
                genotype_size=config['GA']['genotype_size'],
                amount_generations=config['GA']['amount_generations'],
                crossing_rate=config['GA']['crossing_rate'],
                mutation_rate=config['GA']['mutation_rate'],
                mlp=mlp,
                train_data=(X, y),
                val_data=(X_val, y_val))
        ga.build_inital_polulation()
        self.ga = ga
        self.main_window = main_window

    def fit(self):
        print('run_fit')
        self.main_window.button_run_fit.setText("Остановить")
        while (self.ga.number_generation < config['GA']['amount_generations']):
            if self.stop_event.is_set():
                self.stop_event.clear()
                self.main_window.button_run_fit.setText("Запустить")
                break
            scores, hist = self.ga.step_evolution()
            self.main_window.label_generation.setText(str(self.ga.number_generation - 1))
            self.main_window.label_mean_acc.setText(str(np.mean(scores)))
            self.main_window.label_best_acc.setText(str(scores[0]))
            self.main_window.label_deviation.setText(str(np.var(scores)**0.5))

            self.main_window.canvas_fit.data[0] = hist
            self.main_window.canvas_fit.data[1].append(scores[0])
            self.main_window.canvas_fit.data[2].append(np.mean(scores))
            self.main_window.canvas_fit.plot()

def fit_model(main_window):
    m = GA_MLP(main_window)
    m.fit()
