import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from Window1 import Ui_Window1
from mplwidget import resample_x_ticks

from accusleepy.utils.constants import BRAIN_STATE_MAPPER, MAX_LOWER_XTICK_N
from accusleepy.utils.fileio import load_labels, load_recording
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

        self.epochs_to_show = 5

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
        self.upper_spec, self.upper_f = create_spectrogram(
            self.eeg, sampling_rate, epoch_length
        )
        # process, bin, ceiling emg for upper plot
        self.upper_emg = create_upper_emg_signal(self.emg, sampling_rate, epoch_length)
        self.label_img = create_label_img(self.labels, self.label_display_options)

        # set up plots
        self.ui.upperplots.setup_upper_plots(
            self.n_epochs,
            self.label_img,
            self.upper_spec,
            self.upper_f,
            self.upper_emg,
            epoch_length,
            self.epochs_to_show,
            self.label_display_options,
            BRAIN_STATE_MAPPER,
        )
        self.ui.lowerplots.setup_lower_plots(
            self.label_img,
            sampling_rate,
            epoch_length,
            self.epochs_to_show,
            BRAIN_STATE_MAPPER,
            self.label_display_options,
        )

        self.epoch = 0
        self.eeg_signal_scale_factor = 1
        self.emg_signal_scale_factor = 1
        self.eeg_signal_offset = 0
        self.emg_signal_offset = 0
        self.upper_left_epoch = 0
        self.upper_right_epoch = self.n_epochs - 1
        self.lower_left_epoch = 0
        self.lower_right_epoch = self.epochs_to_show - 1
        self.display_labels = convert_labels(self.labels, "display")
        self.process_signals()

        self.update_lower_plot()

        # user input
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

        self.ui.upperplots.canvas.mpl_connect("button_press_event", self.click_to_jump)

        self.autoscroll_state = False
        self.ui.autoscroll.stateChanged.connect(self.update_autoscroll_state)

        self.ui.eegzoomin.clicked.connect(partial(self.update_signal_zoom, "eeg", "in"))
        self.ui.eegzoomout.clicked.connect(
            partial(self.update_signal_zoom, "eeg", "out")
        )
        self.ui.emgzoomin.clicked.connect(partial(self.update_signal_zoom, "emg", "in"))
        self.ui.emgzoomout.clicked.connect(
            partial(self.update_signal_zoom, "emg", "out")
        )

        self.ui.eegshiftup.clicked.connect(
            partial(self.update_signal_offset, "eeg", "up")
        )
        self.ui.eegshiftdown.clicked.connect(
            partial(self.update_signal_offset, "eeg", "down")
        )
        self.ui.emgshiftup.clicked.connect(
            partial(self.update_signal_offset, "emg", "up")
        )
        self.ui.emgshiftdown.clicked.connect(
            partial(self.update_signal_offset, "emg", "down")
        )

        self.ui.shownepochsplus.clicked.connect(
            partial(self.update_epochs_shown, "plus")
        )
        self.ui.shownepochsminus.clicked.connect(
            partial(self.update_epochs_shown, "minus")
        )

        self.show()

    def update_epochs_shown(self, direction: str):
        if direction == "plus":
            self.epochs_to_show += 2
            if self.lower_left_epoch == 0:
                self.lower_right_epoch += 2
            elif self.lower_right_epoch == self.n_epochs - 1:
                self.lower_left_epoch -= 2
            else:
                self.lower_left_epoch -= 1
                self.lower_right_epoch += 1
        else:
            if self.epochs_to_show > 3:
                self.epochs_to_show -= 2
                if self.lower_left_epoch == 0:
                    self.lower_right_epoch -= 2
                elif self.lower_right_epoch == self.n_epochs - 1:
                    self.lower_left_epoch += 2
                else:
                    self.lower_left_epoch += 1
                    self.lower_right_epoch -= 1

        self.ui.shownepochslabel.setText(str(self.epochs_to_show))

        # totally rebuild lower plots
        self.ui.lowerplots.canvas.figure.clf()
        self.ui.lowerplots.setup_lower_plots(
            self.label_img,
            sampling_rate,
            epoch_length,
            self.epochs_to_show,
            BRAIN_STATE_MAPPER,
            self.label_display_options,
        )
        self.update_upper_plot()
        self.update_lower_plot()

    def update_signal_offset(self, signal: str, direction: str):
        offset_increments = {"up": 0.02, "down": -0.02}
        if signal == "eeg":
            self.eeg_signal_offset += offset_increments[direction]
        else:
            self.emg_signal_offset += offset_increments[direction]
        self.update_lower_plot()

    def update_signal_zoom(self, signal: str, direction: str):
        zoom_factors = {"in": 1.08, "out": 0.95}
        if signal == "eeg":
            self.eeg_signal_scale_factor *= zoom_factors[direction]
        else:
            self.emg_signal_scale_factor *= zoom_factors[direction]
        self.update_lower_plot()

    def update_autoscroll_state(self, checked):
        self.autoscroll_state = checked

    def adjust_upper_plot_x_limits(self):
        for i in range(4):
            self.ui.upperplots.canvas.axes[i].set_xlim(
                (self.upper_left_epoch - 0.5, self.upper_right_epoch + 0.5)
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

    def modify_label(self, digit):
        self.labels[self.epoch] = digit
        display_label = convert_labels(
            np.array([digit]),
            style="display",
        )[0]
        self.display_labels[self.epoch] = display_label
        self.modify_label_img(display_label)
        if self.autoscroll_state and self.epoch < self.n_epochs - 1:
            self.shift_epoch("right")

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
        old_window_center = int(self.epochs_to_show / 2) + self.lower_left_epoch
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

    def update_upper_marker(self):
        # this pattern appears elsewhere...
        epoch_padding = int((self.epochs_to_show - 1) / 2)
        if self.epoch - epoch_padding < 0:
            left_edge = 0
            right_edge = self.epochs_to_show - 1
        elif self.epoch + epoch_padding > self.n_epochs - 1:
            right_edge = self.n_epochs - 1
            left_edge = self.n_epochs - self.epochs_to_show
        else:
            left_edge = self.epoch - epoch_padding
            right_edge = self.epoch + epoch_padding

        self.ui.upperplots.upper_marker[0].set_xdata(
            [
                left_edge - 0.5,
                right_edge + 0.5,
            ]
        )
        self.ui.upperplots.upper_marker[1].set_xdata([self.epoch])

    def update_upper_plot(self):
        # WIP
        self.update_upper_marker()
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
        eeg = eeg * self.eeg_signal_scale_factor + self.eeg_signal_offset
        emg = emg * self.emg_signal_scale_factor + self.emg_signal_offset

        self.update_lower_epoch_marker()

        # plot eeg and emg
        self.ui.lowerplots.eeg_line.set_ydata(eeg)
        self.ui.lowerplots.emg_line.set_ydata(emg)

        # plot brain state
        self.ui.lowerplots.label_img_ref.set(
            data=self.label_img[
                :, self.lower_left_epoch : (self.lower_right_epoch + 1), :
            ]
        )
        x_ticks = resample_x_ticks(
            np.arange(self.lower_left_epoch, self.lower_right_epoch + 1)
        )
        self.ui.lowerplots.canvas.axes[1].set_xticklabels(
            [
                "{:02d}:{:02d}:{:05.2f}".format(int(x // 3600), int(x // 60), (x % 60))
                for x in x_ticks * epoch_length
            ]
        )

        self.ui.lowerplots.canvas.draw()

    def click_to_jump(self, event):
        # make sure click location is valid
        if event.xdata is None:
            return
        # get the "zoom level" so we can preserve that
        upper_epochs_shown = self.upper_right_epoch - self.upper_left_epoch + 1
        upper_epoch_padding = int((upper_epochs_shown - 1) / 2)
        # update epoch
        self.epoch = round(np.clip(event.xdata, 0, self.n_epochs - 1))
        # update upper plot x limits
        # find out if the jump puts us too close to either edge
        if self.epoch - upper_epoch_padding < 0:
            self.upper_left_epoch = 0
            self.upper_right_epoch = upper_epochs_shown - 1
        elif self.epoch + upper_epoch_padding > self.n_epochs - 1:
            self.upper_right_epoch = self.n_epochs - 1
            self.upper_left_epoch = self.n_epochs - upper_epochs_shown
        else:
            self.upper_left_epoch = self.epoch - upper_epoch_padding
            self.upper_right_epoch = self.epoch + upper_epoch_padding
        # update upper marker
        self.update_upper_marker()
        # refresh upper plot
        self.adjust_upper_plot_x_limits()

        # update lower plot location
        lower_epoch_padding = int((self.epochs_to_show - 1) / 2)
        if self.epoch - lower_epoch_padding < 0:
            self.lower_left_epoch = 0
            self.lower_right_epoch = self.epochs_to_show - 1
        elif self.epoch + lower_epoch_padding > self.n_epochs - 1:
            self.lower_right_epoch = self.n_epochs - 1
            self.lower_left_epoch = self.n_epochs - self.epochs_to_show
        else:
            self.lower_left_epoch = self.epoch - lower_epoch_padding
            self.lower_right_epoch = self.epoch + lower_epoch_padding

        # refresh lower plot
        self.update_lower_plot()


## EXECUTE APP
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
