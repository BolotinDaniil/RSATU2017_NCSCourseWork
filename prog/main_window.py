import sys
from PyQt5 import QtWidgets, uic

main_window_class = uic.loadUiType('mainwindow.ui')[0]

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random

import numpy as np

from dataset import DataSrc
from config_parser import config
from model import GA_MLP

import threading


class MainWindow(QtWidgets.QMainWindow, main_window_class):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.init_canvases()

        self.data_src = DataSrc()
        self.data_src.load_data('data/RTS_001110_161101.csv',
                                len_group=config['DATASET']['len_group'],
                                len_order=config['DATASET']['len_order'],
                                valid_share=config['DATASET']['valid_share'],
                                test_share=config['DATASET']['test_share'],
                                )

        # show dataset
        data = self.data_src.get_all_raw_data()
        self.canvas_dataset.data = data
        self.canvas_dataset.plot()

        # buttons events
        self.button_view_dataset.clicked.connect(self.button_click_view_dataset)

        self.button_run_fit.clicked.connect(self.button_click_run_fit)
        self.button_view_fit.clicked.connect(self.button_click_view_fit)

        self.button_run_test.clicked.connect(self.button_click_run_test)
        self.button_view_test.clicked.connect(self.button_click_view_test)
        # model
        self.model = GA_MLP(self)
        self.teacher_th = threading.Thread(target=self.model.fit, args=())

    def init_canvases(self):
        m = CanvasDataset(self.tab_dataset, width=7, height=4)
        # self.frame.addWidget(m)
        self.canvas_dataset = m

        self.canvas_fit = CanvasFit(self.tab_fit, width=7, height=3.5)
        self.canvas_test = CanvasTest(self.tab_test, width=7, height=3.5)


    def button_click_view_dataset(self):
        self.canvas_dataset.show_plot()

    def button_click_view_fit(self):
        self.canvas_fit.show_plot()

    def button_click_view_test(self):
        self.canvas_test.show_plot()

    def button_click_run_fit(self):
        # fit_model(self)
        if not self.teacher_th.isAlive():
            self.teacher_th = threading.Thread(target=self.model.fit, args=())
            self.teacher_th.start()
        else:
            self.model.stop_event.set()

    def button_click_run_test(self):
        self.model.test()



class CanvasDataset(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.move(20, 100)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.plot()

    def plot(self):
        ds = len(self.data)
        x = np.arange(0, ds, 1)
        split_point_1 = int(ds * (1 - config['DATASET']['valid_share'] - config['DATASET']['test_share']))
        split_point_2 = int(ds * (1 - config['DATASET']['test_share']))
        # print(split_point_1 , split_point_2, ds)
        # data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(x[:split_point_1], self.data[:split_point_1], 'r-')
        ax.plot(x[split_point_1 - 1 :split_point_2], self.data[split_point_1 - 1 :split_point_2], 'b-')
        ax.plot(x[split_point_2 - 1 :], self.data[split_point_2 - 1:], 'g-')
        ax.set_title('Dataset')
        ax.grid(True)
        self.draw()

    def show_plot(self):
        '''
        show matplotlib window
        :return:
        '''
        ds = len(self.data)
        x = np.arange(0, ds, 1)
        split_point_1 = int(ds * (1 - config['DATASET']['valid_share'] - config['DATASET']['test_share']))
        split_point_2 = int(ds * (1 - config['DATASET']['test_share']))
        # print(split_point_1 , split_point_2, ds)
        # data = [random.random() for i in range(25)]
        ax = plt
        ax.plot(x[:split_point_1], self.data[:split_point_1], 'r-')
        ax.plot(x[split_point_1 - 1:split_point_2], self.data[split_point_1 - 1:split_point_2], 'b-')
        ax.plot(x[split_point_2 - 1:], self.data[split_point_2 - 1:], 'g-')
        ax.title('Dataset')
        ax.grid(True)
        ax.show()

class CanvasFit(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes1 = fig.add_subplot(2, 1, 1)
        self.axes1.set_title('accuracy by fit')
        self.axes2 = fig.add_subplot(2, 1, 2)
        self.axes2.set_title('generations fitness')

        FigureCanvas.__init__(self, fig)
        self.move(20, 180)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.data = [[], [], []]

    def prepare_plot1(self, plot):
        plot.clear()
        plot.set_title('accuracy by fit')
        plot.grid(True)
        plot.plot(self.data[0])

    def prepare_plot2(self, plot):
        plot.clear()
        plot.set_title('generations fitness')
        plot.grid(True)
        plot.plot(self.data[1], 'r')
        plot.plot(self.data[2], 'b')

    def plot(self):
        plot1 = self.figure.add_subplot(2, 1, 1)
        plot2 = self.figure.add_subplot(2, 1, 2)
        self.prepare_plot1(plot1)
        self.prepare_plot2(plot2)

        self.draw()

    def show_plot(self):
        plot1 = plt.subplot(2, 1, 1)
        self.prepare_plot1(plot1)
        plot2 = plt.subplot(2, 1, 2)
        self.prepare_plot2(plot2)

        plt.show()


class CanvasTest(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.move(20, 160)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # raw, true, predict data
        self.data = [[], [], []]

    def prepare_plot(self, plot):
        plot.clear()
        plot.set_title('test')
        plot.grid(True)

        raw, y, pred = self.data
        ds = len(raw)
        x = np.arange(0, ds, 1)
        for i in range(1, ds):
            if abs(pred[i] - y[i]) < 0.5:  # true answer
                color = 'g'
            else:
                color = 'r'
            plot.plot(x[i - 1:i + 1], raw[i - 1:i + 1], color=color)


    def plot(self):
        plot = self.figure.add_subplot(1, 1, 1)
        self.prepare_plot(plot)
        self.draw()

    def show_plot(self):
        plot = plt.subplot(1, 1, 1)
        self.prepare_plot(plot)
        plt.show()





# if __name__ == '__main__':
app = QtWidgets.QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
app.exec_()