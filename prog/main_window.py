import sys
from PyQt5 import QtWidgets, uic

main_window_class = uic.loadUiType('mainwindow.ui')[0]

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random


class MainWindow(QtWidgets.QMainWindow, main_window_class):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        m = PlotCanvas(self.tab_dataset, width=7, height=4)
        # self.frame.addWidget(m)
        m.move(20, 100)
        self.button_view_dataset.clicked.connect(m.plot)



class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        ax.set_title('PyQt Matplotlib Example')
        ax.grid(True)
        self.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    app.exec_()