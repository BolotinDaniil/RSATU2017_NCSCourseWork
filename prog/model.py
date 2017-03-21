import numpy as np
from config_parser import config
from mlp import MLP, DenseLayer, SoftmaxLayer, ReLU
from dataset import DataSrc
from utils import timeit
import theano
import threading

from mlp import MLP, DenseLayer, SoftmaxLayer, ReLU, linear
from keras_mlp import keras_MLP
from genetics import GA
from config_parser import config

class GA_MLP:
    def __init__(self, main_window):
        self.stop_event = threading.Event()
        mlp = self.construct_MLP(config)
        train, valid, test = main_window.data_src.get_dataset()
        ga = GA(population_size=config['GA']['population_size'],
                genotype_size=config['GA']['genotype_size'],
                amount_generations=config['GA']['amount_generations'],
                crossing_rate=config['GA']['crossing_rate'],
                mutation_rate=config['GA']['mutation_rate'],
                mlp=mlp,
                train_data=train,
                val_data=valid)
        ga.build_inital_polulation()
        ga.hist = {}
        ga.hist['all_hist'] = [[], []]
        self.ga = ga
        self.main_window = main_window

    def construct_MLP(self, config):
        layers = []
        for i in range(1, len(config['MLP']['layers']) - 1):
            if config['MLP']['activation_fns'][i-1] == 'relu':
                a_fn = ReLU
            elif config['MLP']['activation_fns'][i-1] == 'Linear':
                a_fn = linear
            else:
                a_fn = ReLU
            layer = DenseLayer(config['MLP']['layers'][i - 1], config['MLP']['layers'][i], a_fn)
            layers.append(layer)
        layer = SoftmaxLayer(config['MLP']['layers'][-2], config['MLP']['layers'][-1])
        layers.append(layer)
        # layers = [DenseLayer(5, 64, ReLU), SoftmaxLayer(64, 2)]
        mlp = keras_MLP(layers, config['MLP']['batch_size'])
        return mlp

    def fit(self):
        print('run_fit')
        self.main_window.button_run_fit.setText("Остановить")
        while (self.ga.number_generation < config['GA']['amount_generations']):
            if self.stop_event.is_set():
                self.stop_event.clear()
                self.main_window.button_run_fit.setText("Запустить")
                break
            scores, hist = self.ga.step_evolution()

            self.ga.hist['tek_generation'] = self.ga.number_generation - 1
            self.ga.hist['mean_acc'] = np.mean(scores)
            self.ga.hist['best_acc'] = scores[0]
            self.ga.hist['deviation'] = np.var(scores)**0.5
            self.ga.hist['cur_hist'] = hist
            self.ga.hist['all_hist'][0].append(scores[0])
            self.ga.hist['all_hist'][1].append(np.mean(scores))

            # show text
            self.main_window.label_generation.setText(str(self.ga.hist['tek_generation']))
            self.main_window.label_mean_acc.setText(str(self.ga.hist['mean_acc']))
            self.main_window.label_best_acc.setText(str(self.ga.hist['best_acc']))
            self.main_window.label_deviation.setText(str(self.ga.hist['deviation']))

            # show plots
            self.main_window.canvas_fit.data[0] = self.ga.hist['cur_hist']
            self.main_window.canvas_fit.data[1] = self.ga.hist['all_hist'][0]
            self.main_window.canvas_fit.data[2] = self.ga.hist['all_hist'][1]
            self.main_window.canvas_fit.plot()

    def test(self):
        train, valid, test = self.main_window.data_src.get_dataset()
        _, raw_valid, raw_test = self.main_window.data_src.get_raw_data()
        X, y = test

        # print(len(pred))
        print(len(y))
        print(len(raw_test))

        spread = self.main_window.double_spin_box_spread.value()
        pred = self.ga.best_nn.predict(X)

        ds = len(y)
        acc = 0
        profit = 0
        for i in range(ds):
            if abs(pred[i] - y[i]) < 0.5:
                acc += 1
                profit += abs(raw_test[i] - raw_test[i - 1])
            else:
                profit -= abs(raw_test[i] - raw_test[i - 1])
            profit -= spread

        acc = float(acc) / ds

        # show result
        self.main_window.label_acc.setText(str(acc))
        self.main_window.label_profit.setText(str(profit))

        self.main_window.canvas_test.data = raw_test, y, pred
        self.main_window.canvas_test.plot()

    def save(self):
        pass

    def load(self):
        pass


# def fit_model(main_window):
#     m = GA_MLP(main_window)
#     m.fit()
