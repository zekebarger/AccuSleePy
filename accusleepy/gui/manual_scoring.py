import random
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PySide6 import QtCore, QtGui, QtWidgets
from Window1 import Ui_Window1

from accusleepy.utils.constants import BRAIN_STATE_MAPPER

# magic spell:
# /Users/zeke/PycharmProjects/AccuSleePy/.venv/lib/python3.13/site-packages/PySide6/Qt/libexec/uic -g python accusleepy/gui/window1.ui -o accusleepy/gui/Window1.py

# https://github.com/RamanLukashevich/Easy_Statistica/blob/main/mplwidget.py


KEY_MAP = {
    QtCore.Qt.Key.Key_Backspace: "backspace",
    QtCore.Qt.Key.Key_Tab: "tab",
    # Add mappings for other keys you need to support
    QtCore.Qt.Key.Key_Return: "enter",
    QtCore.Qt.Key.Key_Enter: "enter",
    QtCore.Qt.Key.Key_Escape: "esc",
    QtCore.Qt.Key.Key_Space: "space",
    QtCore.Qt.Key.Key_End: "end",
    QtCore.Qt.Key.Key_Home: "home",
    QtCore.Qt.Key.Key_Left: "left",
    QtCore.Qt.Key.Key_Up: "up",
    QtCore.Qt.Key.Key_Right: "right",
    QtCore.Qt.Key.Key_Down: "down",
    QtCore.Qt.Key.Key_Delete: "delete",
}


# THESE SHOULD BE ENTERED
sampling_rate = 512
epoch_length = 2.5

# SHOULD BE SET BY USER
epochs_to_show = 5


LABEL_CMAP = np.concatenate(
    [np.array([[1, 1, 1, 1]]), plt.colormaps["tab10"](range(10))], axis=0
)


def convert_labels(labels: np.array, style: str):
    if style == "display":
        return [i if i is not None else -1 for i in labels]
    elif style == "digit":
        return [i if i != -1 else None for i in labels]
    else:
        raise Exception("style must be 'display' or 'digit'")


# MAIN WINDOW CLASS
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_Window1()
        self.ui.setupUi(self)

        self.ui.lowerplots.setup_lower_plots(
            sampling_rate, epochs_to_show, BRAIN_STATE_MAPPER
        )

        # THESE SHOULD BE USER INPUT SOMEHOW
        self.epoch = 0
        self.n_epochs = 25
        self.left_epoch = 0
        self.right_epoch = epochs_to_show - 1

        self.eeg = np.random.random(int(sampling_rate * epoch_length * self.n_epochs))
        self.emg = np.random.random(int(sampling_rate * epoch_length * self.n_epochs))
        self.emg = self.emg**5
        self.labels = np.random.randint(0, 3, self.n_epochs)
        self.display_labels = convert_labels(self.labels, "display")

        self._plot_refs = [None, None]
        self.update_lower_plot()

        keypress_right = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
        keypress_right.activated.connect(partial(self.shift_epoch, 1))
        keypress_right.activated.connect(self.update_lower_plot)

        keypress_left = QtGui.QShortcut(QtGui.QKeySequence("Left"), self)
        keypress_left.activated.connect(partial(self.shift_epoch, -1))
        keypress_left.activated.connect(self.update_lower_plot)

        self.show()

    def get_signal_to_plot(self):
        left = int(self.left_epoch * sampling_rate * epoch_length)
        right = int((self.right_epoch + 1) * sampling_rate * epoch_length)
        return (
            self.eeg[left:right],
            self.emg[left:right],
            self.display_labels[self.left_epoch : (self.right_epoch + 1)],
        )

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

    def update_lower_plot(self):
        eeg, emg, labels = self.get_signal_to_plot()

        # Note: we no longer need to clear the axis.
        if self._plot_refs[0] is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            eeg_refs = self.ui.lowerplots.canvas.axes[0].plot(eeg)
            emg_refs = self.ui.lowerplots.canvas.axes[1].plot(emg)
            self._plot_refs = [eeg_refs[0], emg_refs[0]]
        else:
            self._plot_refs[0].set_ydata(eeg)
            self._plot_refs[1].set_ydata(emg)
        # new data to plot
        for i, label in enumerate(labels):
            self.ui.lowerplots.rectangles[i].set_color(LABEL_CMAP[label + 1])
            self.ui.lowerplots.rectangles[i].set_xy([i, max(label, 0)])

        self.ui.lowerplots.canvas.draw()


## EXECUTE APP
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    # window.show()
    sys.exit(app.exec())
