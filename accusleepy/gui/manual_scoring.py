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

# think about using mplconnect?

KEY_MAP = {
    "backspace": QtCore.Qt.Key.Key_Backspace,
    "right": QtCore.Qt.Key.Key_Right,
    "left": QtCore.Qt.Key.Key_Left,
    "up": QtCore.Qt.Key.Key_Up,
    "down": QtCore.Qt.Key.Key_Down,
    # QtCore.Qt.Key.Key_Tab: "tab",
    # QtCore.Qt.Key.Key_Return: "enter",
    # QtCore.Qt.Key.Key_Enter: "enter",
    # QtCore.Qt.Key.Key_Escape: "esc",
    # QtCore.Qt.Key.Key_Space: "space",
    # QtCore.Qt.Key.Key_End: "end",
    # QtCore.Qt.Key.Key_Home: "home",
    # QtCore.Qt.Key.Key_Delete: "delete",
}


# THESE SHOULD BE ENTERED
sampling_rate = 512
epoch_length = 2.5

# SHOULD BE SET BY USER
epochs_to_show = 5


LABEL_CMAP = np.concatenate(
    [np.array([[1, 1, 1, 1]]), plt.colormaps["tab10"](range(10))], axis=0
)


def convert_labels(labels: np.array, style: str) -> np.array:
    if style == "display":
        # convert 0 to 10, None to 0
        labels = [i if i != 0 else 10 for i in labels]
        return np.array([i if i is not None else 0 for i in labels])
    elif style == "digit":
        # convert 0 to None, 10 to 0
        labels = [i if i != 0 else None for i in labels]
        return np.array([i if i != 10 else 0 for i in labels])
    else:
        raise Exception("style must be 'display' or 'digit'")


# MAIN WINDOW CLASS
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_Window1()
        self.ui.setupUi(self)

        # get set of label options (1-10 range)
        label_display_options = convert_labels(
            np.array([b.digit for b in BRAIN_STATE_MAPPER.brain_states]),
            style="display",
        )

        self.ui.lowerplots.setup_lower_plots(
            sampling_rate,
            epoch_length,
            epochs_to_show,
            BRAIN_STATE_MAPPER,
            label_display_options,
        )

        # THESE SHOULD BE USER INPUT SOMEHOW
        self.n_epochs = 25
        self.eeg = np.random.random(int(sampling_rate * epoch_length * self.n_epochs))
        self.emg = np.random.random(int(sampling_rate * epoch_length * self.n_epochs))
        self.emg = self.emg**5
        self.labels = np.random.randint(1, 4, self.n_epochs)

        self.epoch = 0
        self.eeg_signal_scale_factor = 1
        self.emg_signal_scale_factor = 1
        self.left_epoch = 0
        self.right_epoch = epochs_to_show - 1
        self.display_labels = convert_labels(self.labels, "display")

        self.process_signals()

        # self._plot_refs = [None, None]
        self.update_lower_plot()

        keypress_right = QtGui.QShortcut(QtGui.QKeySequence(KEY_MAP["right"]), self)
        keypress_right.activated.connect(partial(self.shift_epoch, 1))
        keypress_right.activated.connect(self.update_lower_plot)

        keypress_left = QtGui.QShortcut(QtGui.QKeySequence(KEY_MAP["left"]), self)
        keypress_left.activated.connect(partial(self.shift_epoch, -1))
        keypress_left.activated.connect(self.update_lower_plot)

        keypress_modify_label = list()
        for brain_state in BRAIN_STATE_MAPPER.brain_states:
            keypress_modify_label.append(
                QtGui.QShortcut(QtGui.QKeySequence(str(brain_state.digit)), self)
            )
            keypress_modify_label[-1].activated.connect(
                partial(self.modify_label, brain_state.digit)
            )
            keypress_modify_label[-1].activated.connect(self.update_lower_plot)

        # on hold until i figure out how to represent missing labels
        # keypress_delete_label = QtGui.QShortcut(
        #     QtGui.QKeySequence(KEY_MAP["backspace"]), self
        # )
        # keypress_delete_label.activated.connect(partial(self.modify_label, None))
        # keypress_delete_label.activated.connect(self.update_lower_plot)

        self.show()

    # could avoid running this if nothing actually changes...
    def modify_label(self, digit):
        self.labels[self.epoch] = digit
        self.display_labels[self.epoch] = convert_labels(
            np.array([digit]),
            style="display",
        )[0]

    def process_signals(self):
        self.eeg = self.eeg - np.mean(self.eeg)
        self.emg = self.emg - np.mean(self.emg)
        self.eeg = self.eeg / np.percentile(self.eeg, 95) / 2.2
        self.emg = self.emg / np.percentile(self.emg, 95) / 2.2

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
            self.left_epoch -= 1
            self.right_epoch -= 1
        elif self.epoch > old_window_center and self.right_epoch < self.n_epochs - 1:
            self.left_epoch += 1
            self.right_epoch += 1

    def update_lower_epoch_marker(self):
        # plot marker for selected epoch
        marker_left = (self.epoch - self.left_epoch) * epoch_length * sampling_rate
        marker_right = (1 + self.epoch - self.left_epoch) * epoch_length * sampling_rate
        self.ui.lowerplots.top_marker[0].set_xdata([marker_left, marker_left])
        self.ui.lowerplots.top_marker[1].set_xdata([marker_left, marker_right])
        self.ui.lowerplots.top_marker[2].set_xdata([marker_right, marker_right])
        self.ui.lowerplots.bottom_marker[0].set_xdata([marker_left, marker_left])
        self.ui.lowerplots.bottom_marker[1].set_xdata([marker_left, marker_right])
        self.ui.lowerplots.bottom_marker[2].set_xdata([marker_right, marker_right])

    def update_lower_plot(self):
        # get signals to plot
        eeg, emg, labels = self.get_signal_to_plot()
        # zoom in or out
        eeg = eeg * self.eeg_signal_scale_factor
        emg = emg * self.emg_signal_scale_factor

        self.update_lower_epoch_marker()

        # plot eeg and emg
        self.ui.lowerplots.eeg_line.set_ydata(eeg)
        self.ui.lowerplots.emg_line.set_ydata(emg)

        for i, label in enumerate(labels):
            self.ui.lowerplots.rectangles[i].set_color(LABEL_CMAP[label])
            # fix this
            self.ui.lowerplots.rectangles[i].set_xy([i, label])

        self.ui.lowerplots.canvas.draw()


## EXECUTE APP
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    # window.show()
    sys.exit(app.exec())
