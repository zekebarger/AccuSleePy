import sys
import random
from functools import partial

import numpy as np

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

        # THESE SHOULD BE USER INPUT SOMEHOW
        self.epoch = 0
        self.n_epochs = 25
        self.left_epoch = 0
        self.right_epoch = epochs_to_show - 1

        self.eeg = np.random.random(int(sampling_rate * epoch_length * self.n_epochs))
        self.emg = np.random.random(int(sampling_rate * epoch_length * self.n_epochs))
        self.emg = self.emg**5
        self.labels = np.random.randint(0, 3, self.n_epochs)

        self._plot_refs = [None, None]
        self.update_plot()

        keypress_right = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
        keypress_right.activated.connect(partial(self.shift_epoch, 1))
        keypress_right.activated.connect(self.update_plot)

        keypress_left = QtGui.QShortcut(QtGui.QKeySequence("Left"), self)
        keypress_left.activated.connect(partial(self.shift_epoch, -1))
        keypress_left.activated.connect(self.update_plot)

        self.show()

    def get_signal_to_plot(self):
        left = int(self.left_epoch * sampling_rate * epoch_length)
        right = int((self.right_epoch + 1) * sampling_rate * epoch_length)
        return self.eeg[left:right], self.emg[left:right]

    def shift_epoch(self, shift_amount):
        # can't move outside min, max epochs
        if not (0 <= (self.epoch + shift_amount) < self.n_epochs):
            return

        # shift to new epoch
        self.epoch = self.epoch + shift_amount

        old_window_center = int(epochs_to_show / 2) + self.left_epoch
        # change the window bounds if needed
        if self.epoch < old_window_center and self.left_epoch > 0:
            # print("shifting window left")
            self.left_epoch -= 1
            self.right_epoch -= 1
        elif self.epoch > old_window_center and self.right_epoch < self.n_epochs - 1:
            # print("shifting window right")
            self.left_epoch += 1
            self.right_epoch += 1

    def update_plot(self):
        eeg, emg = self.get_signal_to_plot()

        # Note: we no longer need to clear the axis.
        if self._plot_refs[0] is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            eeg_refs = self.ui.mplwidget1.canvas.axes.plot(eeg)
            emg_refs = self.ui.mplwidget2.canvas.axes.plot(emg)
            self._plot_refs = [eeg_refs[0], emg_refs[0]]
        else:
            self._plot_refs[0].set_ydata(eeg)
            self._plot_refs[1].set_ydata(emg)

        self.ui.mplwidget1.canvas.draw()
        self.ui.mplwidget2.canvas.draw()


## EXECUTE APP
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    # window.show()
    sys.exit(app.exec())
