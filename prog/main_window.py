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
from model import fit_model

import threading


class MainWindow(QtWidgets.QMainWindow, main_window_class):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        m = PlotCanvasDataset(self.tab_dataset, width=7, height=4)
        # self.frame.addWidget(m)
        self.plot_canvas_dataset = m

        self.data_src = DataSrc()
        self.data_src.load_raw_data('data/BRENT_001101_161101_1DAY.csv')
        _, _, data = self.data_src.get_raw_data()
        self.plot_canvas_dataset.data = data
        self.plot_canvas_dataset.plot()

        # buttons
        self.button_view_dataset.clicked.connect(self.button_click_view_dataset)
        self.button_run_fit.clicked.connect(self.button_click_run_fit)

    def button_click_view_dataset(self):
        self.plot_canvas_dataset.show_plot()

    def button_click_run_fit(self):
        fit_model(self)

    def button_click_run_test(self):
        pass



class PlotCanvasDataset(FigureCanvas):
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

class PlotCanvasFit(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotCanvasTest(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)




# if __name__ == '__main__':
app = QtWidgets.QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
app.exec_()