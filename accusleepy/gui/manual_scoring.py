# AccuSleePy manual scoring GUI
# Icon sources:
#   Arkinasi, https://www.flaticon.com/authors/arkinasi
#   kendis lasman, https://www.flaticon.com/packs/ui-79

import copy
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from mplwidget import resample_x_ticks
from PySide6 import QtCore, QtGui, QtWidgets
from Window1 import Ui_Window1

from accusleepy.utils.constants import BRAIN_STATE_MAPPER, UNDEFINED_LABEL
from accusleepy.utils.fileio import load_labels, load_recording, save_labels
from accusleepy.utils.misc import SimulatedClick
from accusleepy.utils.signal_processing import create_spectrogram, process_emg

# NOTES
# magic spell:
# /Users/zeke/PycharmProjects/AccuSleePy/.venv/lib/python3.13/site-packages/PySide6/Qt/libexec/uic -g python accusleepy/gui/window1.ui -o accusleepy/gui/Window1.py

# other magic spell (if the icon set is altered)
# pyside6-rcc accusleepy/gui/resources.qrc -o accusleepy/gui/resources_rc.py

# https://www.pythonguis.com/tutorials/qresource-system/
# and if you won't want to mess with qt creator, can add this below customwidgets in the ui file
# <resources>
#   <include location="resources.qrc"/>
#  </resources>


# colormap for displaying brain state labels
# the first entry represents the "undefined" state
# the other entries are the digits in "keyboard" order (1234567890)
LABEL_CMAP = np.concatenate(
    [np.array([[0, 0, 0, 0]]), plt.colormaps["tab10"](range(10))], axis=0
)
# relative path to user manual txt file
USER_MANUAL_FILE = "text/manual1.txt"

# label formats
DISPLAY_FORMAT = "display"
DIGIT_FORMAT = "digit"


# MAIN WINDOW CLASS
class MainWindow(QtWidgets.QMainWindow):
    """AccuSleePy manual scoring GUI"""

    def __init__(self):
        super(MainWindow, self).__init__()

        # THESE SHOULD BE USER INPUT SOMEHOW
        self.label_file = "/Users/zeke/PycharmProjects/AccuSleePy/sample_labels.csv"
        self.eeg, self.emg = load_recording(
            "/Users/zeke/PycharmProjects/AccuSleePy/sample_recording.parquet"
        )
        self.labels = load_labels(self.label_file)
        self.sampling_rate = 512
        self.epoch_length = 2.5

        self.n_epochs = len(self.labels)

        # initialize the UI
        self.ui = Ui_Window1()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy manual scoring window")

        # initial setting for number of epochs to show in the lower plot
        self.epochs_to_show = 5

        # find the set of y-axis locations of valid brain state labels
        self.label_display_options = convert_labels(
            np.array([b.digit for b in BRAIN_STATE_MAPPER.brain_states]),
            style=DISPLAY_FORMAT,
        )
        self.smallest_display_label = np.min(self.label_display_options)

        self.ui.upperplots.epoch_length = self.epoch_length
        self.ui.lowerplots.epoch_length = self.epoch_length

        # get EEG spectrogram and its frequency axis
        spectrogram, spectrogram_frequencies = create_spectrogram(
            self.eeg, self.sampling_rate, self.epoch_length
        )

        # calculate and reformat RMS of EMG for each epoch
        self.upper_emg = create_upper_emg_signal(
            self.emg, self.sampling_rate, self.epoch_length
        )

        # rescale the EEG and EMG signals to fit the display
        self.eeg, self.emg = scale_eeg_emg(self.eeg, self.emg)

        # convert labels to "display" format and make an image to display them
        self.display_labels = convert_labels(self.labels, DISPLAY_FORMAT)
        self.label_img = create_label_img(
            self.display_labels, self.label_display_options
        )

        # set up both figures
        self.ui.upperplots.setup_upper_figure(
            self.n_epochs,
            self.label_img,
            spectrogram,
            spectrogram_frequencies,
            self.upper_emg,
            self.epochs_to_show,
            self.label_display_options,
            BRAIN_STATE_MAPPER,
            self.roi_callback,
        )
        self.ui.lowerplots.setup_lower_figure(
            self.label_img,
            self.sampling_rate,
            self.epochs_to_show,
            BRAIN_STATE_MAPPER,
            self.label_display_options,
        )

        # initialize values that can be changed by user input
        self.epoch = 0
        self.eeg_signal_scale_factor = 1
        self.emg_signal_scale_factor = 1
        self.eeg_signal_offset = 0
        self.emg_signal_offset = 0
        self.upper_left_epoch = 0
        self.upper_right_epoch = self.n_epochs - 1
        self.lower_left_epoch = 0
        self.lower_right_epoch = self.epochs_to_show - 1
        self.roi_brain_state = 0
        self.label_roi_mode = False
        self.autoscroll_state = False
        # keep track of save state to warn user when they quit
        self.last_saved_labels = copy.deepcopy(self.labels)

        # populate the lower figure
        self.update_lower_figure()

        # user input: keyboard shortcuts
        keypress_right = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self
        )
        keypress_right.activated.connect(partial(self.shift_epoch, "right"))
        keypress_right.activated.connect(self.update_lower_figure)
        keypress_right.activated.connect(self.update_upper_figure)

        keypress_left = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self
        )
        keypress_left.activated.connect(partial(self.shift_epoch, "left"))
        keypress_left.activated.connect(self.update_lower_figure)
        keypress_left.activated.connect(self.update_upper_figure)

        keypress_zoom_in_x = list()
        for zoom_key in [QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal]:
            keypress_zoom_in_x.append(
                QtGui.QShortcut(QtGui.QKeySequence(zoom_key), self)
            )
            keypress_zoom_in_x[-1].activated.connect(partial(self.zoom_x, "in"))

        keypress_zoom_out_x = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Minus), self
        )
        keypress_zoom_out_x.activated.connect(partial(self.zoom_x, "out"))

        keypress_modify_label = list()
        for brain_state in BRAIN_STATE_MAPPER.brain_states:
            keypress_modify_label.append(
                QtGui.QShortcut(
                    QtGui.QKeySequence(QtCore.Qt.Key[f"Key_{brain_state.digit}"]),
                    self,
                )
            )
            keypress_modify_label[-1].activated.connect(
                partial(self.modify_current_epoch_label, brain_state.digit)
            )

        keypress_delete_label = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Backspace), self
        )
        keypress_delete_label.activated.connect(
            partial(self.modify_current_epoch_label, UNDEFINED_LABEL)
        )

        keypress_quit = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(QtCore.Qt.Modifier.CTRL, QtCore.Qt.Key.Key_W)
            ),
            self,
        )
        keypress_quit.activated.connect(self.close)

        keypress_save = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(QtCore.Qt.Modifier.CTRL, QtCore.Qt.Key.Key_S)
            ),
            self,
        )
        keypress_save.activated.connect(self.save)

        keypress_roi = list()
        for brain_state in BRAIN_STATE_MAPPER.brain_states:
            keypress_roi.append(
                QtGui.QShortcut(
                    QtGui.QKeySequence(
                        QtCore.QKeyCombination(
                            QtCore.Qt.Modifier.SHIFT,
                            QtCore.Qt.Key[f"Key_{brain_state.digit}"],
                        )
                    ),
                    self,
                )
            )
            keypress_roi[-1].activated.connect(
                partial(self.enter_label_roi_mode, brain_state.digit)
            )
        keypress_roi.append(
            QtGui.QShortcut(
                QtGui.QKeySequence(
                    QtCore.QKeyCombination(
                        QtCore.Qt.Modifier.SHIFT,
                        QtCore.Qt.Key.Key_Backspace,
                    )
                ),
                self,
            )
        )
        keypress_roi[-1].activated.connect(
            partial(self.enter_label_roi_mode, UNDEFINED_LABEL)
        )

        keypress_esc = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self
        )
        keypress_esc.activated.connect(self.exit_label_roi_mode)

        keypress_space = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self
        )
        keypress_space.activated.connect(
            partial(self.jump_to_next_state, "right", "different")
        )
        keypress_shift_right = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(
                    QtCore.Qt.Modifier.SHIFT,
                    QtCore.Qt.Key.Key_Right,
                )
            ),
            self,
        )
        keypress_shift_right.activated.connect(
            partial(self.jump_to_next_state, "right", "different")
        )
        keypress_shift_left = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(
                    QtCore.Qt.Modifier.SHIFT,
                    QtCore.Qt.Key.Key_Left,
                )
            ),
            self,
        )
        keypress_shift_left.activated.connect(
            partial(self.jump_to_next_state, "left", "different")
        )
        keypress_ctrl_right = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(
                    QtCore.Qt.Modifier.CTRL,
                    QtCore.Qt.Key.Key_Right,
                )
            ),
            self,
        )
        keypress_ctrl_right.activated.connect(
            partial(self.jump_to_next_state, "right", "undefined")
        )
        keypress_ctrl_left = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(
                    QtCore.Qt.Modifier.CTRL,
                    QtCore.Qt.Key.Key_Left,
                )
            ),
            self,
        )
        keypress_ctrl_left.activated.connect(
            partial(self.jump_to_next_state, "left", "undefined")
        )

        # user input: clicks
        self.ui.upperplots.canvas.mpl_connect("button_press_event", self.click_to_jump)

        # user input: buttons
        self.ui.savebutton.clicked.connect(self.save)
        self.ui.xzoomin.clicked.connect(partial(self.zoom_x, "in"))
        self.ui.xzoomout.clicked.connect(partial(self.zoom_x, "out"))
        self.ui.xzoomreset.clicked.connect(partial(self.zoom_x, "reset"))
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
        self.ui.specbrighter.clicked.connect(
            partial(self.update_spectrogram_brightness, "brighter")
        )
        self.ui.specdimmer.clicked.connect(
            partial(self.update_spectrogram_brightness, "dimmer")
        )
        self.ui.helpbutton.clicked.connect(self.show_user_manual)

        self.show()

    def closeEvent(self, event):
        if not all(self.labels == self.last_saved_labels):
            result = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "You have unsaved changes. Really quit?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if result == QtWidgets.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

    def show_user_manual(self):
        self.popup = QtWidgets.QWidget()
        self.popup.setGeometry(QtCore.QRect(50, 100, 350, 400))
        grid = QtWidgets.QGridLayout()
        user_manual_file = open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), USER_MANUAL_FILE),
            "r",
        )
        user_manual_text = user_manual_file.read()
        user_manual_file.close()
        label_widget = QtWidgets.QLabel()
        label_widget.setText(user_manual_text)
        grid.addWidget(label_widget)
        self.popup.setLayout(grid)
        self.popup.show()

    def jump_to_next_state(self, direction: str, target: str):
        simulated_click = SimulatedClick(xdata=self.epoch)
        if direction == "right":
            if target == "different":
                matches = np.where(
                    self.labels[self.epoch + 1 :] != self.labels[self.epoch]
                )[0]
            else:
                matches = np.where(self.labels[self.epoch + 1 :] == UNDEFINED_LABEL)[0]
            if matches.size > 0:
                simulated_click.xdata = matches[0] + 1 + self.epoch
        else:
            if target == "different":
                matches = np.where(
                    self.labels[: self.epoch] != self.labels[self.epoch]
                )[0]
            else:
                matches = np.where(self.labels[: self.epoch] == UNDEFINED_LABEL)[0]
            if matches.size > 0:
                simulated_click.xdata = matches[-1]
        self.click_to_jump(simulated_click)

    def roi_callback(self, eclick, erelease):
        self.labels[int(np.ceil(eclick.xdata)) : int(np.floor(erelease.xdata)) + 1] = (
            self.roi_brain_state
        )
        self.display_labels = convert_labels(
            self.labels,
            style=DISPLAY_FORMAT,
        )
        self.label_img = create_label_img(
            self.display_labels, self.label_display_options
        )
        self.update_upper_figure()
        self.update_lower_figure()
        self.exit_label_roi_mode()

    def exit_label_roi_mode(self):
        self.ui.upperplots.roi.set_active(False)
        self.ui.upperplots.roi.set_visible(False)
        self.ui.upperplots.roi.update()
        self.label_roi_mode = False

    def enter_label_roi_mode(self, brain_state):
        self.label_roi_mode = True
        self.roi_brain_state = brain_state
        self.ui.upperplots.roi_patch.set(
            facecolor=LABEL_CMAP[
                convert_labels(np.array([brain_state]), DISPLAY_FORMAT)
            ]
        )
        self.ui.upperplots.roi.set_active(True)

    def save(self):
        save_labels(self.labels, self.label_file)
        self.last_saved_labels = copy.deepcopy(self.labels)

    def update_spectrogram_brightness(self, direction: str):
        vmin, vmax = self.ui.upperplots.spec_ref.get_clim()
        if direction == "brighter":
            self.ui.upperplots.spec_ref.set(clim=(vmin, vmax * 0.96))
        else:
            self.ui.upperplots.spec_ref.set(clim=(vmin, vmax * 1.07))
        self.ui.upperplots.canvas.draw()

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
        self.ui.lowerplots.setup_lower_figure(
            self.label_img,
            self.sampling_rate,
            self.epochs_to_show,
            BRAIN_STATE_MAPPER,
            self.label_display_options,
        )
        self.update_upper_figure()
        self.update_lower_figure()

    def update_signal_offset(self, signal: str, direction: str):
        offset_increments = {"up": 0.02, "down": -0.02}
        if signal == "eeg":
            self.eeg_signal_offset += offset_increments[direction]
        else:
            self.emg_signal_offset += offset_increments[direction]
        self.update_lower_figure()

    def update_signal_zoom(self, signal: str, direction: str):
        zoom_factors = {"in": 1.08, "out": 0.95}
        if signal == "eeg":
            self.eeg_signal_scale_factor *= zoom_factors[direction]
        else:
            self.emg_signal_scale_factor *= zoom_factors[direction]
        self.update_lower_figure()

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
        elif direction == "out":
            self.upper_left_epoch = int(max([0, self.epoch - 1.017 * epochs_shown]))
            self.upper_right_epoch = int(
                min([self.n_epochs - 1, self.epoch + 1.017 * epochs_shown])
            )
        else:  # reset
            self.upper_left_epoch = 0
            self.upper_right_epoch = self.n_epochs - 1
        self.adjust_upper_plot_x_limits()

    def modify_label_img(self, display_label):
        if display_label == 0:  # undefined brain state
            self.label_img[:, self.epoch] = np.array([0, 0, 0, 1])
        else:
            self.label_img[:, self.epoch, :] = 1
            self.label_img[
                display_label - self.smallest_display_label, self.epoch, :
            ] = LABEL_CMAP[display_label]

    def modify_current_epoch_label(self, digit):
        self.labels[self.epoch] = digit
        display_label = convert_labels(
            np.array([digit]),
            style=DISPLAY_FORMAT,
        )[0]
        self.display_labels[self.epoch] = display_label
        self.modify_label_img(display_label)
        if self.autoscroll_state and self.epoch < self.n_epochs - 1:
            self.shift_epoch("right")
        self.update_upper_figure()
        self.update_lower_figure()

    def get_signal_to_plot(self):
        left = int(self.lower_left_epoch * self.sampling_rate * self.epoch_length)
        right = int(
            (self.lower_right_epoch + 1) * self.sampling_rate * self.epoch_length
        )
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

    def update_upper_figure(self):
        self.update_upper_marker()
        self.ui.upperplots.label_img_ref.set(data=self.label_img)
        self.ui.upperplots.canvas.draw()

    def update_lower_epoch_marker(self):
        # plot marker for selected epoch
        marker_left = (
            (self.epoch - self.lower_left_epoch)
            * self.epoch_length
            * self.sampling_rate
        )
        marker_right = (
            (1 + self.epoch - self.lower_left_epoch)
            * self.epoch_length
            * self.sampling_rate
        )
        self.ui.lowerplots.top_marker[0].set_xdata([marker_left, marker_left])
        self.ui.lowerplots.top_marker[1].set_xdata([marker_left, marker_right])
        self.ui.lowerplots.top_marker[2].set_xdata([marker_right, marker_right])
        self.ui.lowerplots.bottom_marker[0].set_xdata([marker_left, marker_left])
        self.ui.lowerplots.bottom_marker[1].set_xdata([marker_left, marker_right])
        self.ui.lowerplots.bottom_marker[2].set_xdata([marker_right, marker_right])

    def update_lower_figure(self):
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
                for x in x_ticks * self.epoch_length
            ]
        )

        self.ui.lowerplots.canvas.draw()

    def click_to_jump(self, event):
        # make sure click location is valid
        # and we are not in label ROI mode
        if event.xdata is None or self.label_roi_mode:
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
        self.update_lower_figure()


def convert_labels(labels: np.array, style: str) -> np.array:
    if style == DISPLAY_FORMAT:
        # convert 0 to 10, undefined to 0
        labels = [i if i != 0 else 10 for i in labels]
        return np.array([i if i != UNDEFINED_LABEL else 0 for i in labels])
    elif style == DIGIT_FORMAT:
        # convert 0 to undefined, 10 to 0
        labels = [i if i != 0 else UNDEFINED_LABEL for i in labels]
        return np.array([i if i != 10 else 0 for i in labels])
    else:
        raise Exception(f"style must be '{DISPLAY_FORMAT}' or '{DIGIT_FORMAT}'")


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
        else:
            label_img[:, i] = np.array([0, 0, 0, 1])
    return label_img


def create_upper_emg_signal(emg, sampling_rate, epoch_length):
    binned_emg = process_emg(
        emg,
        sampling_rate,
        epoch_length,
    )
    emg_ceiling = np.mean(binned_emg) + np.std(binned_emg) * 2.5
    binned_emg[binned_emg > emg_ceiling] = emg_ceiling
    return binned_emg


def scale_eeg_emg(eeg: np.array, emg: np.array) -> (np.array, np.array):
    eeg = eeg - np.mean(eeg)
    emg = emg - np.mean(emg)
    eeg = eeg / np.percentile(eeg, 95) / 2.2
    emg = emg / np.percentile(emg, 95) / 2.2
    return eeg, emg


## EXECUTE APP
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
