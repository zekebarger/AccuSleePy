import random
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PySide6 import QtCore, QtGui, QtWidgets
from Window1 import Ui_Window1

from accusleepy.utils.constants import BRAIN_STATE_MAPPER
from accusleepy.utils.fileio import load_recording, load_labels
from accusleepy.utils.signal_processing import create_spectrogram, process_emg

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
    "control": QtCore.Qt.Key.Key_Control,
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


def create_upper_emg_signal(emg, sampling_rate, epoch_length):
    binned_emg = process_emg(
        emg,
        sampling_rate,
        epoch_length,
    )
    emg_ceiling = np.mean(binned_emg) + np.std(binned_emg) * 2.5
    binned_emg[binned_emg > emg_ceiling] = emg_ceiling
    return binned_emg


def create_label_img(labels, label_display_options):
    smallest_display_label = np.min(label_display_options)
    label_img = np.ones(
        [
            (np.max(label_display_options) - smallest_display_label + 1),
            len(labels),
            4,
        ]
    )
    for i, label in enumerate(labels):
        if label > 0:
            label_img[label - smallest_display_label, i, :] = LABEL_CMAP[label]
    return label_img


# MAIN WINDOW CLASS
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # THESE SHOULD BE USER INPUT SOMEHOW
        self.eeg, self.emg = load_recording(
            "/Users/zeke/PycharmProjects/AccuSleePy/sample_recording.parquet"
        )
        self.labels = load_labels(
            "/Users/zeke/PycharmProjects/AccuSleePy/sample_labels.csv"
        )

        self.ui = Ui_Window1()
        self.ui.setupUi(self)

        # get set of label options (1-10 range)
        self.label_display_options = convert_labels(
            np.array([b.digit for b in BRAIN_STATE_MAPPER.brain_states]),
            style="display",
        )
        self.smallest_display_label = np.min(self.label_display_options)

        self.n_epochs = len(self.labels)
        # process data to show in the upper plot - we just change x limits
        upper_spec, upper_f = create_spectrogram(self.eeg, sampling_rate, epoch_length)
        # process, bin, ceiling emg for upper plot
        upper_emg = create_upper_emg_signal(self.emg, sampling_rate, epoch_length)
        self.label_img = create_label_img(self.labels, self.label_display_options)

        # set up plots
        self.ui.upperplots.setup_upper_plots(
            self.n_epochs,
            self.label_img,
            upper_spec,
            upper_f,
            upper_emg,
            epochs_to_show,
            self.label_display_options,
            BRAIN_STATE_MAPPER,
        )
        self.ui.lowerplots.setup_lower_plots(
            sampling_rate,
            epoch_length,
            epochs_to_show,
            BRAIN_STATE_MAPPER,
            self.label_display_options,
        )

        self.epoch = 0
        self.eeg_signal_scale_factor = 1
        self.emg_signal_scale_factor = 1
        self.upper_left_epoch = 0
        self.upper_right_epoch = self.n_epochs - 1
        self.lower_left_epoch = 0
        self.lower_right_epoch = epochs_to_show - 1
        self.display_labels = convert_labels(self.labels, "display")
        self.process_signals()

        self.update_lower_plot()

        keypress_right = QtGui.QShortcut(QtGui.QKeySequence(KEY_MAP["right"]), self)
        keypress_right.activated.connect(partial(self.shift_epoch, "right"))
        keypress_right.activated.connect(self.update_lower_plot)
        keypress_right.activated.connect(self.update_upper_plot)

        keypress_left = QtGui.QShortcut(QtGui.QKeySequence(KEY_MAP["left"]), self)
        keypress_left.activated.connect(partial(self.shift_epoch, "left"))
        keypress_left.activated.connect(self.update_lower_plot)
        keypress_left.activated.connect(self.update_upper_plot)

        # set these to plus and minus??
        keypress_zoom_in_x = QtGui.QShortcut(QtGui.QKeySequence("i"), self)
        keypress_zoom_in_x.activated.connect(partial(self.zoom_x, "in"))
        keypress_zoom_out_x = QtGui.QShortcut(QtGui.QKeySequence("o"), self)
        keypress_zoom_out_x.activated.connect(partial(self.zoom_x, "out"))

        keypress_modify_label = list()
        for brain_state in BRAIN_STATE_MAPPER.brain_states:
            keypress_modify_label.append(
                QtGui.QShortcut(QtGui.QKeySequence(str(brain_state.digit)), self)
            )
            keypress_modify_label[-1].activated.connect(
                partial(self.modify_label, brain_state.digit)
            )
            # this should be optimized!
            keypress_modify_label[-1].activated.connect(self.update_upper_plot)
            keypress_modify_label[-1].activated.connect(self.update_lower_plot)

        # on hold until I figure out how to represent missing labels
        # keypress_delete_label = QtGui.QShortcut(
        #     QtGui.QKeySequence(KEY_MAP["backspace"]), self
        # )
        # keypress_delete_label.activated.connect(partial(self.modify_label, None))
        # keypress_delete_label.activated.connect(self.update_lower_plot)

        keypress_quit = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+w"), self)
        keypress_quit.activated.connect(lambda: self.close())

        self.show()

    def adjust_upper_plot_x_limits(self):
        for i in range(4):
            self.ui.upperplots.canvas.axes[i].set_xlim(
                (self.upper_left_epoch, self.upper_right_epoch)
            )
        self.ui.upperplots.canvas.draw()

    def zoom_x(self, direction: str):
        epochs_shown = self.upper_right_epoch - self.upper_left_epoch + 1
        # is int too imprecise?
        if direction == "in":
            self.upper_left_epoch = int(
                max([self.upper_left_epoch, self.epoch - 0.45 * epochs_shown])
            )
            self.upper_right_epoch = int(
                min([self.upper_right_epoch, self.epoch + 0.45 * epochs_shown])
            )
        else:
            self.upper_left_epoch = int(max([0, self.epoch - 1.017 * epochs_shown]))
            self.upper_right_epoch = int(
                min([self.n_epochs - 1, self.epoch + 1.017 * epochs_shown])
            )
        self.adjust_upper_plot_x_limits()

    def modify_label_img(self, display_label):
        self.label_img[:, self.epoch, :] = 1
        self.label_img[display_label - self.smallest_display_label, self.epoch, :] = (
            LABEL_CMAP[display_label]
        )

    # could avoid running this if nothing actually changes...
    def modify_label(self, digit):
        self.labels[self.epoch] = digit
        display_label = convert_labels(
            np.array([digit]),
            style="display",
        )[0]
        self.display_labels[self.epoch] = display_label
        self.modify_label_img(display_label)

    def process_signals(self):
        self.eeg = self.eeg - np.mean(self.eeg)
        self.emg = self.emg - np.mean(self.emg)
        self.eeg = self.eeg / np.percentile(self.eeg, 95) / 2.2
        self.emg = self.emg / np.percentile(self.emg, 95) / 2.2

    def get_signal_to_plot(self):
        left = int(self.lower_left_epoch * sampling_rate * epoch_length)
        right = int((self.lower_right_epoch + 1) * sampling_rate * epoch_length)
        return (
            self.eeg[left:right],
            self.emg[left:right],
            self.display_labels[self.lower_left_epoch : (self.lower_right_epoch + 1)],
        )

    def shift_upper_plots(self, direction: str):
        # update upper plot if needed
        upper_epochs_shown = self.upper_right_epoch - self.upper_left_epoch + 1
        if (
            self.epoch > self.upper_left_epoch + 0.65 * upper_epochs_shown
            and self.upper_right_epoch < (self.n_epochs - 1)
            and direction == "right"
        ):
            self.upper_left_epoch += 1
            self.upper_right_epoch += 1
            self.adjust_upper_plot_x_limits()
        elif (
            self.epoch < self.upper_left_epoch + 0.35 * upper_epochs_shown
            and self.upper_left_epoch > 0
            and direction == "left"
        ):
            self.upper_left_epoch -= 1
            self.upper_right_epoch -= 1
            self.adjust_upper_plot_x_limits()

    def shift_epoch(self, direction: str):
        shift_amount = {"left": -1, "right": 1}[direction]
        # can't move outside min, max epochs
        if not (0 <= (self.epoch + shift_amount) < self.n_epochs):
            return

        # shift to new epoch
        self.epoch = self.epoch + shift_amount

        self.shift_upper_plots(direction)

        # update parts of lower plot
        old_window_center = int(epochs_to_show / 2) + self.lower_left_epoch
        # change the window bounds if needed
        if self.epoch < old_window_center and self.lower_left_epoch > 0:
            self.lower_left_epoch -= 1
            self.lower_right_epoch -= 1
        elif (
            self.epoch > old_window_center
            and self.lower_right_epoch < self.n_epochs - 1
        ):
            self.lower_left_epoch += 1
            self.lower_right_epoch += 1

    def shift_upper_marker(self):
        self.ui.upperplots.upper_marker[0].set_xdata(
            [self.lower_left_epoch, self.lower_right_epoch + 1]
        )
        self.ui.upperplots.upper_marker[1].set_xdata([self.epoch])

    def update_upper_plot(self):
        # WIP
        self.shift_upper_marker()
        self.ui.upperplots.label_img_ref.set(data=self.label_img)
        self.ui.upperplots.canvas.draw()

    def update_lower_epoch_marker(self):
        # plot marker for selected epoch
        marker_left = (
            (self.epoch - self.lower_left_epoch) * epoch_length * sampling_rate
        )
        marker_right = (
            (1 + self.epoch - self.lower_left_epoch) * epoch_length * sampling_rate
        )
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

        # just use the image here too!! lkjhasdfkljasdfjhk
        for i, label in enumerate(labels):
            self.ui.lowerplots.rectangles[i].set_color(LABEL_CMAP[label])
            self.ui.lowerplots.rectangles[i].set_xy([i, label])

        self.ui.lowerplots.canvas.draw()


## EXECUTE APP
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
