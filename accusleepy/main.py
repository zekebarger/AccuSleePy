import sys
import random
import matplotlib

# from IPython.external.qt_for_kernel import QtGui

matplotlib.use("QtAgg")
from PySide6 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


KEY_REPEAT_DELAY = 0.75
KEY_REPEAT_SPEED = 0.1


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setGeometry(100, 100, 1000, 800)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # self.setCentralWidget(self.canvas)

        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [random.randint(0, 10) for i in range(n_data)]

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self._plot_ref = None
        self.update_plot()

        self.button = QtWidgets.QPushButton("Plot")
        self.button.clicked.connect(self.update_plot)

        # set the layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(
            self.canvas,
            0,
            0,
            2,
            1,
        )
        layout.addWidget(self.button, 1, 0, 2, 1)

        self.key_label = QtWidgets.QLabel("Last Key Pressed: None", self.canvas)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()

        # # Setup a timer to trigger the redraw by calling update_plot.
        # self.timer = QtCore.QTimer()
        # self.timer.setInterval(100)
        # self.timer.timeout.connect(self.update_plot)
        # self.timer.start()

    def keyPressEvent(self, event):
        if isinstance(event, QtGui.QKeyEvent):
            key_text = event.text()
            self.key_label.setText(f"Last Key Pressed: {key_text}")

    def keyReleaseEvent(self, event):
        if isinstance(event, QtGui.QKeyEvent):
            key_text = event.text()
            self.key_label.setText(f"Key Released: {key_text}")

    def update_plot(self):
        # Drop off the first y element, append a new one.
        self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        # Note: we no longer need to clear the axis.
        if self._plot_ref is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_refs = self.canvas.axes.plot(self.xdata, self.ydata, "r")
            self._plot_ref = plot_refs[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref.set_ydata(self.ydata)

        # Trigger the canvas to update and redraw.
        self.canvas.draw()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec()
