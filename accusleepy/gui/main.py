# AccuSleePy main window

import os
import sys

import numpy as np
from primary_window import Ui_PrimaryWindow
from PySide6 import QtCore, QtGui, QtWidgets

from accusleepy.gui.manual_scoring import ManualScoringWindow
from accusleepy.utils.classification import (create_calibration_file,
                                             score_recording)
from accusleepy.utils.constants import (BRAIN_STATE_MAPPER, EPOCHS_PER_IMG,
                                        MIXTURE_MEAN_COL, MIXTURE_SD_COL,
                                        UNDEFINED_LABEL)
from accusleepy.utils.fileio import (load_calibration_file, load_labels,
                                     load_model, load_recording, save_labels)
from accusleepy.utils.misc import Recording, enforce_min_bout_length
from accusleepy.utils.signal_processing import resample_and_standardize

# max number of messages to display
MESSAGE_BOX_MAX_DEPTH = 50
LABEL_LENGTH_ERROR = "label file length does not match recording length"
# relative path to user manual txt file
USER_MANUAL_FILE = "text/main_manual.txt"


class AccuSleepWindow(QtWidgets.QMainWindow):
    """AccuSleePy main window"""

    def __init__(self):
        super(AccuSleepWindow, self).__init__()

        # initialize the UI
        self.ui = Ui_PrimaryWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy")

        # initialize info about the recordings, classification data / settings
        self.epoch_length = 0
        self.calibration_data = {MIXTURE_MEAN_COL: None, MIXTURE_SD_COL: None}
        self.model = None
        self.only_overwrite_undefined = False
        self.min_bout_length = 5

        # set up the list of recordings
        first_recording = Recording(
            widget=QtWidgets.QListWidgetItem(
                "Recording 1", self.ui.recording_list_widget
            ),
        )
        self.ui.recording_list_widget.addItem(first_recording.widget)
        self.ui.recording_list_widget.setCurrentRow(0)
        # index of currently selected recording in the list
        self.recording_index = 0
        # list of recordings the user has added
        self.recordings = [first_recording]

        # messages to display
        self.messages = []

        # user input: keyboard shortcuts
        keypress_quit = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(QtCore.Qt.Modifier.CTRL, QtCore.Qt.Key.Key_W)
            ),
            self,
        )
        keypress_quit.activated.connect(self.close)

        # user input: button presses
        self.ui.add_button.clicked.connect(self.add_recording)
        self.ui.remove_button.clicked.connect(self.remove_recording)
        self.ui.recording_list_widget.currentRowChanged.connect(self.select_recording)
        self.ui.sampling_rate_input.valueChanged.connect(self.update_sampling_rate)
        self.ui.epoch_length_input.valueChanged.connect(self.update_epoch_length)
        self.ui.recording_file_button.clicked.connect(self.select_recording_file)
        self.ui.select_label_button.clicked.connect(self.select_label_file)
        self.ui.create_label_button.clicked.connect(self.create_label_file)
        self.ui.manual_scoring_button.clicked.connect(self.manual_scoring)
        self.ui.create_calibration_button.clicked.connect(self.create_calibration_file)
        self.ui.load_calibration_button.clicked.connect(self.load_calibration_file)
        self.ui.load_model_button.clicked.connect(self.load_model)
        self.ui.score_all_button.clicked.connect(self.score_all)
        self.ui.overwritecheckbox.stateChanged.connect(self.update_overwrite_policy)
        self.ui.bout_length_input.valueChanged.connect(self.update_min_bout_length)
        self.ui.user_manual_button.clicked.connect(self.show_user_manual)

        self.show()

    def score_all(self) -> None:
        """Score all recordings using the classification model"""
        # check basic inputs
        if self.calibration_data[MIXTURE_MEAN_COL] is None:
            self.ui.score_all_status.setText("missing calibration file")
            self.show_message("ERROR: no calibration file selected")
            return
        if self.model is None:
            self.ui.score_all_status.setText("missing classification model")
            self.show_message("ERROR: no classification model file selected")
            return
        if self.min_bout_length < self.epoch_length:
            self.ui.score_all_status.setText("invalid minimum bout length")
            self.show_message("ERROR: minimum bout length must be >= epoch length")
            return

        self.ui.score_all_status.setText("running...")
        self.ui.score_all_status.repaint()
        app.processEvents()

        # check some inputs for each recording
        for recording_index in range(len(self.recordings)):
            error_message = self.check_single_file_inputs(recording_index)
            if error_message:
                self.ui.score_all_status.setText(
                    f"error on recording {self.recordings[recording_index].name}"
                )
                self.show_message(
                    f"ERROR ({self.recordings[recording_index].name}): {error_message}"
                )
                return

        # score each recording
        for recording_index in range(len(self.recordings)):
            try:
                eeg, emg = load_recording(
                    self.recordings[recording_index].recording_file
                )
                sampling_rate = self.recordings[recording_index].sampling_rate

                eeg, emg, sampling_rate = resample_and_standardize(
                    eeg=eeg,
                    emg=emg,
                    sampling_rate=sampling_rate,
                    epoch_length=self.epoch_length,
                )
            except Exception:
                self.show_message(
                    (
                        "ERROR: could not load recording "
                        f"{self.recordings[recording_index].name}."
                        "This recording will be skipped."
                    )
                )
                continue

            label_file = self.recordings[recording_index].label_file
            if os.path.isfile(label_file):
                try:
                    existing_labels = load_labels(label_file)
                except Exception:
                    self.show_message(
                        (
                            "ERROR: could not load existing labels for recording "
                            f"{self.recordings[recording_index].name}."
                            "This recording will be skipped."
                        )
                    )
                    continue
                # only check the length
                samples_per_epoch = sampling_rate * self.epoch_length
                epochs_in_recording = int(eeg.size / samples_per_epoch)
                if epochs_in_recording != existing_labels.size:
                    self.show_message(
                        (
                            "ERROR: existing labels for recording "
                            f"{self.recordings[recording_index].name} "
                            "do not match the recording length. "
                            "This recording will be skipped."
                        )
                    )
                    continue
            else:
                existing_labels = None

            labels = score_recording(
                model=self.model,
                eeg=eeg,
                emg=emg,
                mixture_means=self.calibration_data[MIXTURE_MEAN_COL],
                mixture_sds=self.calibration_data[MIXTURE_SD_COL],
                sampling_rate=sampling_rate,
                epoch_length=self.epoch_length,
                epochs_per_img=EPOCHS_PER_IMG,
            )

            # overwrite as needed
            if existing_labels is not None and self.only_overwrite_undefined:
                labels[existing_labels != UNDEFINED_LABEL] = existing_labels[
                    existing_labels != UNDEFINED_LABEL
                ]

            # enforce minimum bout length
            labels = enforce_min_bout_length(
                labels=labels,
                epoch_length=self.epoch_length,
                min_bout_length=self.min_bout_length,
            )

            # save results
            save_labels(labels, label_file)
            self.show_message(
                (
                    "Saved labels for recording "
                    f"{self.recordings[recording_index].name} "
                    f"to {label_file}"
                )
            )

        self.ui.score_all_status.setText("")

    def load_model(self) -> None:
        """Load trained classification model from file"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select classification model")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("*.pth")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            if not os.path.isfile(filename):
                self.show_message("ERROR: model file does not exist")
                return
            try:
                self.model = load_model(filename)
            except Exception:
                self.show_message(
                    (
                        "ERROR: could not load classification model. Check "
                        "user manual for instructions on creating this file."
                    )
                )
                return

            self.ui.model_label.setText(filename)

    def load_calibration_file(self) -> None:
        """Load calibration data from file"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select calibration file")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("*.csv")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            if not os.path.isfile(filename):
                self.show_message("ERROR: calibration file does not exist")
                return
            try:
                (
                    self.calibration_data[MIXTURE_MEAN_COL],
                    self.calibration_data[MIXTURE_SD_COL],
                ) = load_calibration_file(filename)
            except Exception:
                self.show_message(
                    (
                        "ERROR: could not load calibration file. Check user "
                        "manual for instructions on creating this file."
                    )
                )
                return

            self.ui.calibration_file_label.setText(filename)

    def load_single_recording(
        self, status_widget: QtWidgets.QLabel
    ) -> (np.array, np.array, int | float, bool):
        """Load and preprocess one recording

        This loads one recording, resamples it, and standardizes its length.
        If an error occurs during this process, it is displayed in the
        indicated widget.

        :param status_widget: UI element on which to display error messages
        :return: EEG data, EMG data, sampling rate, process completion
        """
        error_message = self.check_single_file_inputs(self.recording_index)
        if error_message:
            status_widget.setText(error_message)
            self.show_message(f"ERROR: {error_message}")
            return None, None, None, False

        try:
            eeg, emg = load_recording(
                self.recordings[self.recording_index].recording_file
            )
        except Exception:
            status_widget.setText("could not load recording")
            self.show_message(
                (
                    "ERROR: could not load recording. "
                    "Check user manual for formatting instructions."
                )
            )
            return None, None, None, False

        sampling_rate = self.recordings[self.recording_index].sampling_rate

        eeg, emg, sampling_rate = resample_and_standardize(
            eeg=eeg,
            emg=emg,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
        )

        return eeg, emg, sampling_rate, True

    def create_calibration_file(self) -> None:
        """Creates a calibration file

        This loads a recording and its labels, checks that the labels are
        all valid, creates the calibration file, and makes the contents of
        the calibration file available to the main window.
        """
        # load the recording
        eeg, emg, sampling_rate, success = self.load_single_recording(
            self.ui.calibration_status
        )
        if not success:
            return

        # load the labels
        label_file = self.recordings[self.recording_index].label_file
        if not os.path.isfile(label_file):
            self.ui.calibration_status.setText("label file does not exist")
            self.show_message("ERROR: label file does not exist")
            return
        try:
            labels = load_labels(label_file)
        except Exception:
            self.ui.calibration_status.setText("could not load labels")
            self.show_message(
                (
                    "ERROR: could not load labels. "
                    "Check user manual for formatting instructions."
                )
            )
            return
        label_error_message = check_label_validity(
            labels=labels,
            samples_in_recording=eeg.size,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
        )
        if label_error_message:
            self.ui.calibration_status.setText("invalid label file")
            self.show_message(f"ERROR: {label_error_message}")
            return

        # get the name for the calibration file
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Save calibration file as",
            filter="*.csv",
        )
        if not filename:
            return

        create_calibration_file(
            filename=filename,
            eeg=eeg,
            emg=emg,
            labels=labels,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
        )

        self.ui.calibration_status.setText("")
        self.show_message(
            (
                "Created calibration file using recording "
                f"{self.recordings[self.recording_index].name} "
                f"at {filename}"
            )
        )

        # get the contents of the calibration file
        (
            self.calibration_data[MIXTURE_MEAN_COL],
            self.calibration_data[MIXTURE_SD_COL],
        ) = load_calibration_file(filename)
        self.ui.calibration_file_label.setText(filename)

    def check_single_file_inputs(self, recording_index: int) -> str:
        """Check that a recording's inputs appear valid

        This runs some basic tests for whether it will be possible to
        load and score a recording. If any test fails, we return an
        error message.

        :param recording_index: index of the recording in the list of
            all recordings.
        :return: error message
        """
        sampling_rate = self.recordings[recording_index].sampling_rate
        if self.epoch_length == 0:
            return "epoch length can't be 0"
        if sampling_rate == 0:
            return "sampling rate can't be 0"
        if self.epoch_length > sampling_rate:
            return "invalid epoch length or sampling rate"
        if self.recordings[self.recording_index].recording_file == "":
            return "no recording selected"
        if self.recordings[self.recording_index].label_file == "":
            return "no label file selected"

    def update_min_bout_length(self, new_value) -> None:
        """Update the minimum bout length

        :param new_value: new minimum bout length, in seconds
        """
        self.min_bout_length = new_value

    def update_overwrite_policy(self, checked) -> None:
        """Toggle overwriting policy

        If the checkbox is enabled, only epochs where the brain state is set to
        undefined will be overwritten by the automatic scoring process.

        :param checked: state of the checkbox
        """
        self.only_overwrite_undefined = checked

    def manual_scoring(self) -> None:
        """View the selected recording for manual scoring"""
        # immediately display a status message
        self.ui.manual_scoring_status.setText("loading...")
        self.ui.manual_scoring_status.repaint()
        app.processEvents()

        # load the recording
        eeg, emg, sampling_rate, success = self.load_single_recording(
            self.ui.calibration_status
        )
        if not success:
            return

        # if the labels exist, load them
        # otherwise, create a blank set of labels
        label_file = self.recordings[self.recording_index].label_file
        if os.path.isfile(label_file):
            try:
                labels = load_labels(label_file)
            except Exception:
                self.ui.manual_scoring_status.setText("could not load labels")
                self.show_message(
                    (
                        "ERROR: could not load labels. "
                        "Check user manual for formatting instructions."
                    )
                )
                return
        else:
            labels = (
                np.ones(int(eeg.size / (sampling_rate * self.epoch_length)))
                * UNDEFINED_LABEL
            ).astype(int)

        # check that all labels are valid
        label_error = check_label_validity(
            labels=labels,
            samples_in_recording=eeg.size,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
        )
        if label_error:
            # if the label length is only off by one, pad or truncate as needed
            # and show a warning
            if label_error == LABEL_LENGTH_ERROR:
                samples_per_epoch = sampling_rate * self.epoch_length
                epochs_in_recording = int(eeg.size / samples_per_epoch)
                if epochs_in_recording - labels.size == 1:
                    labels = np.concatenate((labels, np.array([UNDEFINED_LABEL])))
                    self.show_message(
                        (
                            "WARNING: an undefined epoch was added to "
                            "the label file to correct its length."
                        )
                    )
                elif labels.size - epochs_in_recording == 1:
                    labels = labels[:-1]
                    self.show_message(
                        (
                            "WARNING: the last epoch was removed from "
                            "the label file to correct its length."
                        )
                    )
                else:
                    self.ui.manual_scoring_status.setText("invalid label file")
                    self.show_message(f"ERROR: {label_error}")
                    return
            else:
                self.ui.manual_scoring_status.setText("invalid label file")
                self.show_message(f"ERROR: {label_error}")
                return

        self.show_message(
            f"Viewing recording {self.recordings[self.recording_index].name}"
        )
        self.ui.manual_scoring_status.setText("file is open")

        # launch the manual scoring window
        manual_scoring_window = ManualScoringWindow(
            eeg=eeg,
            emg=emg,
            label_file=label_file,
            labels=labels,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
        )
        manual_scoring_window.setWindowTitle(f"AccuSleePy viewer: {label_file}")
        manual_scoring_window.exec()
        self.ui.manual_scoring_status.setText("")

    def create_label_file(self) -> None:
        """Set the filename for a new label file"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Set filename for label file (nothing will be overwritten yet)",
            filter="*.csv",
        )
        if filename:
            self.recordings[self.recording_index].label_file = filename
            self.ui.label_file_label.setText(filename)

    def select_label_file(self) -> None:
        """User can select an existing label file"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select label file")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("(*.csv)")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            self.recordings[self.recording_index].label_file = filename
            self.ui.label_file_label.setText(filename)

    def select_recording_file(self) -> None:
        """User can select a recording file"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select recording file")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("(*.parquet *.csv)")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            self.recordings[self.recording_index].recording_file = filename
            self.ui.recording_file_label.setText(filename)

    def show_recording_info(self) -> None:
        """Update the UI to show info for the selected recording"""
        self.ui.sampling_rate_input.setValue(
            self.recordings[self.recording_index].sampling_rate
        )
        self.ui.recording_file_label.setText(
            self.recordings[self.recording_index].recording_file
        )
        self.ui.label_file_label.setText(
            self.recordings[self.recording_index].label_file
        )

    def update_epoch_length(self, new_value: int | float) -> None:
        """Update the epoch length when the widget state changes

        :param new_value: new epoch length
        """
        self.epoch_length = new_value

    def update_sampling_rate(self, new_value: int | float) -> None:
        """Update recording's sampling rate when the widget state changes

        :param new_value: new sampling rate
        """
        self.recordings[self.recording_index].sampling_rate = new_value

    def show_message(self, message: str) -> None:
        """Display a new message to the user

        :param message: message to display
        """
        self.messages.append(message)
        if len(self.messages) > MESSAGE_BOX_MAX_DEPTH:
            del self.messages[0]
        self.ui.message_area.setText("\n".join(self.messages))
        # scroll to the bottom
        scrollbar = self.ui.message_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def select_recording(self, list_index: int) -> None:
        """Callback for when a recording is selected

        :param list_index: index of this recording in the list widget
        """
        # get index of this recording
        self.recording_index = list_index
        # display information about this recording
        self.show_recording_info()
        self.ui.selected_recording_groupbox.setTitle(
            (
                "Data / actions for the selected recording "
                f"(Recording {self.recordings[list_index].name}) "
                "from this subject)"
            )
        )

    def add_recording(self) -> None:
        """Add new recording to the list"""
        # find name to use for the new recording
        new_name = max([r.name for r in self.recordings]) + 1

        # add new recording to list
        self.recordings.append(
            Recording(
                name=new_name,
                sampling_rate=self.recordings[self.recording_index].sampling_rate,
                widget=QtWidgets.QListWidgetItem(
                    f"Recording {new_name}", self.ui.recording_list_widget
                ),
            )
        )

        # display new list
        self.ui.recording_list_widget.addItem(self.recordings[-1].widget)
        self.ui.recording_list_widget.setCurrentRow(len(self.recordings) - 1)
        self.show_message(f"added Recording {new_name}")

    def remove_recording(self) -> None:
        """Delete selected recording from the list"""
        if len(self.recordings) > 1:
            current_list_index = self.ui.recording_list_widget.currentRow()
            _ = self.ui.recording_list_widget.takeItem(current_list_index)
            self.show_message(
                f"deleted Recording {self.recordings[current_list_index].name}"
            )
            del self.recordings[current_list_index]

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
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(label_widget)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(scroll_area)
        self.popup = QtWidgets.QWidget()
        self.popup.setLayout(grid)
        self.popup.setGeometry(QtCore.QRect(100, 100, 600, 600))
        self.popup.show()


def check_label_validity(
    labels: np.array,
    samples_in_recording: int,
    sampling_rate: int | float,
    epoch_length: int | float,
) -> str:
    """Check whether a set of brain state labels is valid

    This returns an error message if a problem is found with the
    brain state labels.

    :param labels: brain state labels
    :param samples_in_recording: number of samples in the recording
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :return: error message
    """
    # check that length is correct
    samples_per_epoch = sampling_rate * epoch_length
    epochs_in_recording = int(samples_in_recording / samples_per_epoch)
    if epochs_in_recording != labels.size:
        return LABEL_LENGTH_ERROR

    # check that entries are valid
    if not set(labels.tolist()).issubset(
        set([b.digit for b in BRAIN_STATE_MAPPER.brain_states] + [UNDEFINED_LABEL])
    ):
        return "label file contains invalid entries"


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AccuSleepWindow()
    sys.exit(app.exec())
