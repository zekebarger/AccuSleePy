import sys
import random
from functools import partial

from PySide6 import QtCore, QtWidgets, QtGui
from Window1 import Ui_Window1


# magic spell:
# /Users/zeke/PycharmProjects/AccuSleePy/.venv/lib/python3.13/site-packages/PySide6/Qt/libexec/uic -g python accusleepy/gui/window1.ui -o accusleepy/gui/Window1.py


# THESE SHOULD BE ENTERED
sampling_rate = 512
epoch_length = 2.5

# SHOULD BE SET BY USER
epochs_to_show = 5


# MAIN WINDOW CLASS
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_Window1()
        self.ui.setupUi(self)

        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [random.randint(0, 10) for i in range(n_data)]
        self._plot_ref = None
        self.update_plot()

        keypress_right = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
        keypress_right.activated.connect(partial(self.update_plot, "right"))

        keypress_left = QtGui.QShortcut(QtGui.QKeySequence("Left"), self)
        keypress_left.activated.connect(partial(self.update_plot, "left"))

        self.show()

    def update_plot(self, direction="right"):
        # print("updating")
        if direction == "left":
            self.ydata = [random.randint(0, 10)] + self.ydata[:-1]
        else:
            self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        # Note: we no longer need to clear the axis.
        if self._plot_ref is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_refs = self.ui.mplwidget1.canvas.axes.plot(self.xdata, self.ydata, "r")
            self._plot_ref = plot_refs[0]
        else:
            self._plot_ref.set_ydata(self.ydata)

        self.ui.mplwidget1.canvas.draw()


## EXECUTE APP
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    # window.show()
    sys.exit(app.exec())
