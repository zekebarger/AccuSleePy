# AccuSleePy manual scoring GUI
# Icon sources:
#   Arkinasi, https://www.flaticon.com/authors/arkinasi
#   kendis lasman, https://www.flaticon.com/packs/ui-79

import copy
import os
import sys
from functools import partial
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from mplwidget import resample_x_ticks
from PySide6 import QtCore, QtGui, QtWidgets
from viewer_window import Ui_ViewerWindow

from accusleepy.config import UNDEFINED_LABEL
from accusleepy.fileio import load_config, load_labels, load_recording, save_labels
from accusleepy.signal_processing import (
    create_spectrogram,
    process_emg,
    resample_and_standardize,
)


# colormap for displaying brain state labels
# the first entry represents the "undefined" state
# the other entries are the digits in "keyboard" order (1234567890)
LABEL_CMAP = np.concatenate(
    [np.array([[0, 0, 0, 0]]), plt.colormaps["tab10"](range(10))], axis=0
)
# relative path to user manual txt file
USER_MANUAL_FILE = "text/manual1.txt"

# constants used by callback functions
# label formats
DISPLAY_FORMAT = "display"
DIGIT_FORMAT = "digit"
# offset changes
OFFSET_UP = "up"
OFFSET_DOWN = "down"
OFFSET_INCREMENTS = {OFFSET_UP: 0.02, OFFSET_DOWN: -0.02}
# changes to number of epochs
DIRECTION_PLUS = "plus"
DIRECTION_MINUS = "minus"
# changes to selected epoch
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"
# zoom directions
ZOOM_IN = "in"
ZOOM_OUT = "out"
ZOOM_RESET = "reset"
SIGNAL_ZOOM_FACTORS = {ZOOM_IN: 1.08, ZOOM_OUT: 0.95}
# signal names
EEG_SIGNAL = "eeg"
EMG_SIGNAL = "emg"
# spectrogram color changes
BRIGHTER = "brighter"
DIMMER = "dimmer"
# next epoch target
DIFFERENT_STATE = "different"
UNDEFINED_STATE = "undefined"
# how far from the edge of the upper plot the marker should be
# before starting to scroll again - must be in (0, 0.5)
SCROLL_BOUNDARY = 0.35


class ManualScoringWindow(QtWidgets.QDialog):
    """AccuSleePy manual scoring GUI"""

    def __init__(
        self,
        eeg: np.array,
        emg: np.array,
        label_file: str,
        labels: np.array,
        sampling_rate: int | float,
        epoch_length: int | float,
    ):
        """Initialize the manual scoring window

        :param eeg: EEG signal
        :param emg: EMG signal
        :param label_file: filename for labels
        :param labels: brain state labels
        :param sampling_rate: sampling rate, in Hz
        :param epoch_length: epoch length, in seconds
        """
        super(ManualScoringWindow, self).__init__()

        self.label_file = label_file
        self.eeg = eeg
        self.emg = emg
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length

        self.n_epochs = len(self.labels)

        # initialize the UI
        self.ui = Ui_ViewerWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy manual scoring window")

        # load set of valid brain states
        self.brain_state_set = load_config()

        # initial setting for number of epochs to show in the lower plot
        self.epochs_to_show = 5

        # find the set of y-axis locations of valid brain state labels
        self.label_display_options = convert_labels(
            np.array([b.digit for b in self.brain_state_set.brain_states]),
            style=DISPLAY_FORMAT,
        )
        self.smallest_display_label = np.min(self.label_display_options)

        self.ui.upperfigure.epoch_length = self.epoch_length
        self.ui.lowerfigure.epoch_length = self.epoch_length

        # get EEG spectrogram and its frequency axis
        spectrogram, spectrogram_frequencies = create_spectrogram(
            self.eeg, self.sampling_rate, self.epoch_length
        )

        # calculate RMS of EMG for each epoch and apply a ceiling
        self.upper_emg = create_upper_emg_signal(
            self.emg, self.sampling_rate, self.epoch_length
        )

        # center and scale the EEG and EMG signals to fit the display
        self.eeg, self.emg = transform_eeg_emg(self.eeg, self.emg)

        # convert labels to "display" format and make an image to display them
        self.display_labels = convert_labels(self.labels, DISPLAY_FORMAT)
        self.label_img = create_label_img(
            self.display_labels, self.label_display_options
        )

        # set up both figures
        self.ui.upperfigure.setup_upper_figure(
            self.n_epochs,
            self.label_img,
            spectrogram,
            spectrogram_frequencies,
            self.upper_emg,
            self.epochs_to_show,
            self.label_display_options,
            self.brain_state_set,
            self.roi_callback,
        )
        self.ui.lowerfigure.setup_lower_figure(
            self.label_img,
            self.sampling_rate,
            self.epochs_to_show,
            self.brain_state_set,
            self.label_display_options,
        )

        # initialize values that can be changed by user input
        self.epoch = 0
        self.upper_left_epoch = 0
        self.upper_right_epoch = self.n_epochs - 1
        self.lower_left_epoch = 0
        self.lower_right_epoch = self.epochs_to_show - 1
        self.eeg_signal_scale_factor = 1
        self.emg_signal_scale_factor = 1
        self.eeg_signal_offset = 0
        self.emg_signal_offset = 0
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
        keypress_right.activated.connect(partial(self.shift_epoch, DIRECTION_RIGHT))

        keypress_left = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self
        )
        keypress_left.activated.connect(partial(self.shift_epoch, DIRECTION_LEFT))

        keypress_zoom_in_x = list()
        for zoom_key in [QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal]:
            keypress_zoom_in_x.append(
                QtGui.QShortcut(QtGui.QKeySequence(zoom_key), self)
            )
            keypress_zoom_in_x[-1].activated.connect(partial(self.zoom_x, ZOOM_IN))

        keypress_zoom_out_x = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Minus), self
        )
        keypress_zoom_out_x.activated.connect(partial(self.zoom_x, ZOOM_OUT))

        keypress_modify_label = list()
        for brain_state in self.brain_state_set.brain_states:
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
        for brain_state in self.brain_state_set.brain_states:
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
            partial(self.jump_to_next_state, DIRECTION_RIGHT, DIFFERENT_STATE)
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
            partial(self.jump_to_next_state, DIRECTION_RIGHT, DIFFERENT_STATE)
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
            partial(self.jump_to_next_state, DIRECTION_LEFT, DIFFERENT_STATE)
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
            partial(self.jump_to_next_state, DIRECTION_RIGHT, UNDEFINED_STATE)
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
            partial(self.jump_to_next_state, DIRECTION_LEFT, UNDEFINED_STATE)
        )

        # user input: clicks
        self.ui.upperfigure.canvas.mpl_connect("button_press_event", self.click_to_jump)

        # user input: buttons
        self.ui.savebutton.clicked.connect(self.save)
        self.ui.xzoomin.clicked.connect(partial(self.zoom_x, ZOOM_IN))
        self.ui.xzoomout.clicked.connect(partial(self.zoom_x, ZOOM_OUT))
        self.ui.xzoomreset.clicked.connect(partial(self.zoom_x, ZOOM_RESET))
        self.ui.autoscroll.stateChanged.connect(self.update_autoscroll_state)
        self.ui.eegzoomin.clicked.connect(
            partial(self.update_signal_zoom, EEG_SIGNAL, ZOOM_IN)
        )
        self.ui.eegzoomout.clicked.connect(
            partial(self.update_signal_zoom, EEG_SIGNAL, ZOOM_OUT)
        )
        self.ui.emgzoomin.clicked.connect(
            partial(self.update_signal_zoom, EMG_SIGNAL, ZOOM_IN)
        )
        self.ui.emgzoomout.clicked.connect(
            partial(self.update_signal_zoom, EMG_SIGNAL, ZOOM_OUT)
        )
        self.ui.eegshiftup.clicked.connect(
            partial(self.update_signal_offset, EEG_SIGNAL, OFFSET_UP)
        )
        self.ui.eegshiftdown.clicked.connect(
            partial(self.update_signal_offset, EEG_SIGNAL, OFFSET_DOWN)
        )
        self.ui.emgshiftup.clicked.connect(
            partial(self.update_signal_offset, EMG_SIGNAL, OFFSET_UP)
        )
        self.ui.emgshiftdown.clicked.connect(
            partial(self.update_signal_offset, EMG_SIGNAL, OFFSET_DOWN)
        )
        self.ui.shownepochsplus.clicked.connect(
            partial(self.update_epochs_shown, DIRECTION_PLUS)
        )
        self.ui.shownepochsminus.clicked.connect(
            partial(self.update_epochs_shown, DIRECTION_MINUS)
        )
        self.ui.specbrighter.clicked.connect(
            partial(self.update_spectrogram_brightness, BRIGHTER)
        )
        self.ui.specdimmer.clicked.connect(
            partial(self.update_spectrogram_brightness, DIMMER)
        )
        self.ui.helpbutton.clicked.connect(self.show_user_manual)

        self.show()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Check if there are unsaved changes before closing"""
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

    def show_user_manual(self) -> None:
        """Show a popup window with the user manual"""
        user_manual_file = open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), USER_MANUAL_FILE),
            "r",
        )
        user_manual_text = user_manual_file.read()
        user_manual_file.close()
        label_widget = QtWidgets.QLabel()
        label_widget.setText(user_manual_text)

        self.popup = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout()
        grid.addWidget(label_widget)
        self.popup.setLayout(grid)
        self.popup.setGeometry(QtCore.QRect(50, 100, 350, 400))
        self.popup.show()

    def jump_to_next_state(self, direction: str, target: str) -> None:
        """Jump to epoch based on a target brain state

        This allows the user to jump to the next epoch in a given direction
        (left or right) that has a given state (undefined, or different from
        the current epoch). It's useful for reviewing state transitions or
        locating unlabeled epochs.

        :param direction: left or right
        :param target: different or undefined
        """
        # create a simulated click so we can reuse click_to_jump
        simulated_click = SimpleNamespace(**{"xdata": self.epoch})
        if direction == DIRECTION_RIGHT:
            if target == DIFFERENT_STATE:
                matches = np.where(
                    self.labels[self.epoch + 1 :] != self.labels[self.epoch]
                )[0]
            else:
                matches = np.where(self.labels[self.epoch + 1 :] == UNDEFINED_LABEL)[0]
            if matches.size > 0:
                simulated_click.xdata = int(matches[0]) + 1 + self.epoch
        else:
            if target == DIFFERENT_STATE:
                matches = np.where(
                    self.labels[: self.epoch] != self.labels[self.epoch]
                )[0]
            else:
                matches = np.where(self.labels[: self.epoch] == UNDEFINED_LABEL)[0]
            if matches.size > 0:
                simulated_click.xdata = int(matches[-1])
        self.click_to_jump(simulated_click)

    def roi_callback(self, eclick, erelease) -> None:
        """Callback for ROI labeling widget

        This is called by the RectangleSelector widget when the user finishes
        drawing an ROI. It sets a range of epochs to the desired brain state.
        The function signature is required to have this format.
        """
        # update all three representations of the labels
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
        # update the plots
        self.update_figures()
        self.exit_label_roi_mode()

    def exit_label_roi_mode(self) -> None:
        """Restore the normal GUI state after an ROI is drawn"""
        self.ui.upperfigure.roi.set_active(False)
        self.ui.upperfigure.roi.set_visible(False)
        self.ui.upperfigure.roi.update()
        self.label_roi_mode = False

    def enter_label_roi_mode(self, brain_state) -> None:
        """Enter ROI drawing mode

        In this mode, a user can draw an ROI on the upper brain state label
        image to set a range of epochs to a new brain state.

        :param brain_state: new brain state to set
        """
        self.label_roi_mode = True
        self.roi_brain_state = brain_state
        self.ui.upperfigure.roi_patch.set(
            facecolor=LABEL_CMAP[
                convert_labels(np.array([brain_state]), DISPLAY_FORMAT)
            ]
        )
        self.ui.upperfigure.roi.set_active(True)

    def save(self) -> None:
        """Save brain state labels to file"""
        save_labels(self.labels, self.label_file)
        self.last_saved_labels = copy.deepcopy(self.labels)

    def update_spectrogram_brightness(self, direction: str) -> None:
        """Modify spectrogram color range based on button press

        :param direction: brighter or dimmer
        """
        vmin, vmax = self.ui.upperfigure.spec_ref.get_clim()
        if direction == BRIGHTER:
            self.ui.upperfigure.spec_ref.set(clim=(vmin, vmax * 0.96))
        else:
            self.ui.upperfigure.spec_ref.set(clim=(vmin, vmax * 1.07))
        self.ui.upperfigure.canvas.draw()

    def update_epochs_shown(self, direction: str) -> None:
        """Change the number of epochs shown based on button press

        The user can change the number of epochs shown in the lower figure
        via button presses. This requires extensive changes to both figures.
        The number of epochs can only change in increments of 2 and should
        always be an odd number >= 3.

        :param direction: plus or minus
        """
        # if we are near the beginning or end of the recording, we need
        # to change the epoch range differently.
        if direction == DIRECTION_PLUS:
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

        # rebuild lower figure from scratch
        self.ui.lowerfigure.canvas.figure.clf()
        self.ui.lowerfigure.setup_lower_figure(
            self.label_img,
            self.sampling_rate,
            self.epochs_to_show,
            self.brain_state_set,
            self.label_display_options,
        )
        self.update_figures()

    def update_signal_offset(self, signal: str, direction: str) -> None:
        """Shift EEG or EMG up or down

        :param signal: eeg or emg
        :param direction: up or down
        """
        if signal == EEG_SIGNAL:
            self.eeg_signal_offset += OFFSET_INCREMENTS[direction]
        else:
            self.emg_signal_offset += OFFSET_INCREMENTS[direction]
        self.update_lower_figure()

    def update_signal_zoom(self, signal: str, direction: str) -> None:
        """Zoom EEG or EMG y-axis

        :param signal: eeg or emg
        :param direction: in or out
        """
        if signal == EEG_SIGNAL:
            self.eeg_signal_scale_factor *= SIGNAL_ZOOM_FACTORS[direction]
        else:
            self.emg_signal_scale_factor *= SIGNAL_ZOOM_FACTORS[direction]
        self.update_lower_figure()

    def update_autoscroll_state(self, checked) -> None:
        """Toggle autoscroll behavior

        If autoscroll is enabled, setting the brain state of the current epoch
        via a keypress will advance to the next epoch.

        :param checked: state of the checkbox
        """
        self.autoscroll_state = checked

    def adjust_upper_figure_x_limits(self) -> None:
        """Update the x-axis limits of the upper figure subplots"""
        for i in range(4):
            self.ui.upperfigure.canvas.axes[i].set_xlim(
                (self.upper_left_epoch - 0.5, self.upper_right_epoch + 0.5)
            )
        self.ui.upperfigure.canvas.draw()

    def zoom_x(self, direction: str) -> None:
        """Change upper figure x-axis zoom level

        :param direction: in, out, or reset
        """
        zoom_in_factor = 0.45
        zoom_out_factor = 1.017
        epochs_shown = self.upper_right_epoch - self.upper_left_epoch + 1
        if direction == ZOOM_IN:
            self.upper_left_epoch = int(
                max([self.upper_left_epoch, self.epoch - zoom_in_factor * epochs_shown])
            )
            self.upper_right_epoch = int(
                min(
                    [self.upper_right_epoch, self.epoch + zoom_in_factor * epochs_shown]
                )
            )
        elif direction == ZOOM_OUT:
            self.upper_left_epoch = int(
                max([0, self.epoch - zoom_out_factor * epochs_shown])
            )
            self.upper_right_epoch = int(
                min([self.n_epochs - 1, self.epoch + zoom_out_factor * epochs_shown])
            )
        else:  # reset
            self.upper_left_epoch = 0
            self.upper_right_epoch = self.n_epochs - 1
        self.adjust_upper_figure_x_limits()

    def modify_current_epoch_label(self, digit: int) -> None:
        """Change the current epoch's brain state label

        :param digit: new brain state label in "digit" format
        """
        self.labels[self.epoch] = digit
        display_label = convert_labels(
            np.array([digit]),
            style=DISPLAY_FORMAT,
        )[0]
        self.display_labels[self.epoch] = display_label
        # update the label image
        if display_label == 0:
            self.label_img[:, self.epoch] = np.array([0, 0, 0, 1])
        else:
            self.label_img[:, self.epoch, :] = 1
            self.label_img[
                display_label - self.smallest_display_label, self.epoch, :
            ] = LABEL_CMAP[display_label]
        # autoscroll, if that is enabled
        if self.autoscroll_state and self.epoch < self.n_epochs - 1:
            self.shift_epoch(DIRECTION_RIGHT)  # this calls update_figures()
        else:
            self.update_figures()

    def shift_epoch(self, direction: str) -> None:
        """Set the current epoch one step forward or backward

        When the user presses the left or right arrow key, the previous
        or next epoch will be selected. There are a variety of edge cases
        that need to be handled separately for the upper and lower figures.

        :param direction: left or right
        """
        shift_amount = {DIRECTION_LEFT: -1, DIRECTION_RIGHT: 1}[direction]
        # prevent movement outside the data range
        if not (0 <= (self.epoch + shift_amount) < self.n_epochs):
            return

        # shift to new epoch
        self.epoch = self.epoch + shift_amount

        # update upper plot if needed
        upper_epochs_shown = self.upper_right_epoch - self.upper_left_epoch + 1
        if (
            self.epoch
            > self.upper_left_epoch + (1 - SCROLL_BOUNDARY) * upper_epochs_shown
            and self.upper_right_epoch < (self.n_epochs - 1)
            and direction == DIRECTION_RIGHT
        ):
            self.upper_left_epoch += 1
            self.upper_right_epoch += 1
            self.adjust_upper_figure_x_limits()
        elif (
            self.epoch < self.upper_left_epoch + SCROLL_BOUNDARY * upper_epochs_shown
            and self.upper_left_epoch > 0
            and direction == DIRECTION_LEFT
        ):
            self.upper_left_epoch -= 1
            self.upper_right_epoch -= 1
            self.adjust_upper_figure_x_limits()

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

        self.update_figures()

    def update_upper_marker(self) -> None:
        """Update location of the upper figure's epoch marker"""
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
        self.ui.upperfigure.upper_marker[0].set_xdata(
            [
                left_edge - 0.5,
                right_edge + 0.5,
            ]
        )
        self.ui.upperfigure.upper_marker[1].set_xdata([self.epoch])

    def update_lower_epoch_marker(self) -> None:
        """Update location of the lower figure's epoch marker"""
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
        self.ui.lowerfigure.top_marker[0].set_xdata([marker_left, marker_left])
        self.ui.lowerfigure.top_marker[1].set_xdata([marker_left, marker_right])
        self.ui.lowerfigure.top_marker[2].set_xdata([marker_right, marker_right])
        self.ui.lowerfigure.bottom_marker[0].set_xdata([marker_left, marker_left])
        self.ui.lowerfigure.bottom_marker[1].set_xdata([marker_left, marker_right])
        self.ui.lowerfigure.bottom_marker[2].set_xdata([marker_right, marker_right])

    def update_figures(self) -> None:
        """Update and redraw both figures"""
        # upper figure
        self.update_upper_marker()
        # this step isn't always needed, but it's not too expensive
        self.ui.upperfigure.label_img_ref.set(data=self.label_img)
        self.ui.upperfigure.canvas.draw()
        # lower figure
        self.update_lower_figure()

    def update_lower_figure(self) -> None:
        """Update and redraw the lower figure"""
        # get subset of signals to plot
        first_sample = int(
            self.lower_left_epoch * self.sampling_rate * self.epoch_length
        )
        last_sample = int(
            (self.lower_right_epoch + 1) * self.sampling_rate * self.epoch_length
        )
        eeg = self.eeg[first_sample:last_sample]
        emg = self.emg[first_sample:last_sample]

        # scale and shift as needed
        eeg = eeg * self.eeg_signal_scale_factor + self.eeg_signal_offset
        emg = emg * self.emg_signal_scale_factor + self.emg_signal_offset

        self.update_lower_epoch_marker()

        # replot eeg and emg
        self.ui.lowerfigure.eeg_line.set_ydata(eeg)
        self.ui.lowerfigure.emg_line.set_ydata(emg)

        # replot brain state
        self.ui.lowerfigure.label_img_ref.set(
            data=self.label_img[
                :, self.lower_left_epoch : (self.lower_right_epoch + 1), :
            ]
        )
        # update timestamps
        x_ticks = resample_x_ticks(
            np.arange(self.lower_left_epoch, self.lower_right_epoch + 1)
        )
        self.ui.lowerfigure.canvas.axes[1].set_xticklabels(
            [
                "{:02d}:{:02d}:{:05.2f}".format(int(x // 3600), int(x // 60), (x % 60))
                for x in x_ticks * self.epoch_length
            ]
        )

        self.ui.lowerfigure.canvas.draw()

    def click_to_jump(self, event) -> None:
        """Jump to a new epoch when the user clicks on the upper figure

        This is the callback for mouse clicks on the upper figure. Clicking on
        any of the subplots will jump to the nearest epoch.

        :param event: a MouseEvent containing the click data
        """
        # make sure click location is valid
        # and we are not in label ROI mode
        if event.xdata is None or self.label_roi_mode:
            return
        # get the "zoom level" so we can preserve that
        upper_epochs_shown = self.upper_right_epoch - self.upper_left_epoch + 1
        upper_epoch_padding = int((upper_epochs_shown - 1) / 2)
        # update epoch
        self.epoch = round(np.clip(event.xdata, 0, self.n_epochs - 1))
        # update upper figure x-axis limits
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
        self.adjust_upper_figure_x_limits()

        # update lower figure x-axis range
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

        self.update_lower_figure()


def convert_labels(labels: np.array, style: str) -> np.array:
    """Convert labels between "display" and "digit" formats

    It's useful to represent brain state labels in two ways:
    Digit format: this is how labels are represented in files. It matches the digit
        attribute of the BrainState class as well as the number pressed on the
        keyboard to set an epoch to that brain state.
    Display format: the y-axis value associated with a brain state when brain state
        labels are displayed as an image. This is also the index of the brain state
        in the colormap. Undefined epochs are mapped to 0, and digits are mapped to
        the numbers 1-10 in keyboard order (1234567890).

    :param labels: brain state labels
    :param style: target format for the output
    :return: formatted labels
    """
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


def create_label_img(labels: np.array, label_display_options: np.array) -> np.array:
    """Create an image to display brain state labels

    :param labels: brain state labels, in "display" format
    :param label_display_options: y-axis locations of valid brain state labels
    :return: brain state label image
    """
    # While there can be up to 10 valid brain states, it's possible that not all of them
    # are in use. We don't need to display brain states below and above the range of
    # valid brain states, since those rows would always be empty.
    smallest_display_label = np.min(label_display_options)
    # "background" of the image is white
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
            # label is undefined
            label_img[:, i] = np.array([0, 0, 0, 1])
    return label_img


def create_upper_emg_signal(
    emg: np.array, sampling_rate: int | float, epoch_length: int | float
) -> np.array:
    """Calculate RMS of EMG for each epoch and apply a ceiling

    :param emg: EMG signal
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :return: processed EMG signal
    """
    emg_rms = process_emg(
        emg,
        sampling_rate,
        epoch_length,
    )
    return np.clip(emg_rms, 0, np.mean(emg_rms) + np.std(emg_rms) * 2.5)


def transform_eeg_emg(eeg: np.array, emg: np.array) -> (np.array, np.array):
    """Center and scale the EEG and EMG signals

    A heuristic approach to fitting the EEG and EMG signals in the plot.

    :param eeg: EEG signal
    :param emg: EMG signal
    :return: centered and scaled signals
    """
    eeg = eeg - np.mean(eeg)
    emg = emg - np.mean(emg)
    eeg = eeg / np.percentile(eeg, 95) / 2.2
    emg = emg / np.percentile(emg, 95) / 2.2
    return eeg, emg
