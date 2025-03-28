# AccuSleePy main window
# Icon source: Arkinasi, https://www.flaticon.com/authors/arkinasi

import datetime
import os
import shutil
import sys
from dataclasses import dataclass
from functools import partial

import numpy as np
from primary_window import Ui_PrimaryWindow
from PySide6 import QtCore, QtGui, QtWidgets

from accusleepy.brain_state_set import BrainState, BrainStateSet
from accusleepy.classification import (
    create_calibration_file,
    score_recording,
    train_model,
)
from accusleepy.constants import (
    CALIBRATION_FILE_TYPE,
    DEFAULT_MODEL_TYPE,
    LABEL_FILE_TYPE,
    MODEL_FILE_TYPE,
    RECORDING_FILE_TYPES,
    UNDEFINED_LABEL,
)
from accusleepy.fileio import (
    Recording,
    load_calibration_file,
    load_config,
    load_labels,
    load_model,
    load_recording,
    save_config,
    save_labels,
    save_model,
)
from accusleepy.gui.manual_scoring import ManualScoringWindow
from accusleepy.signal_processing import (
    ANNOTATIONS_FILENAME,
    create_training_images,
    enforce_min_bout_length,
    resample_and_standardize,
)

# max number of messages to display
MESSAGE_BOX_MAX_DEPTH = 50
LABEL_LENGTH_ERROR = "label file length does not match recording length"
# relative path to user manual txt file
USER_MANUAL_FILE = "text/main_guide.txt"
CONFIG_GUIDE_FILE = "text/config_guide.txt"


@dataclass
class StateSettings:
    """Widgets for config settings for a brain state"""

    digit: int
    enabled_widget: QtWidgets.QCheckBox
    name_widget: QtWidgets.QLabel
    is_scored_widget: QtWidgets.QCheckBox
    frequency_widget: QtWidgets.QDoubleSpinBox


class AccuSleepWindow(QtWidgets.QMainWindow):
    """AccuSleePy primary window"""

    def __init__(self):
        super(AccuSleepWindow, self).__init__()

        # initialize the UI
        self.ui = Ui_PrimaryWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy")

        # fill in settings tab
        self.brain_state_set = load_config()
        self.settings_widgets = None
        self.initialize_settings_tab()

        # initialize info about the recordings, classification data / settings
        self.epoch_length = 0
        self.model = None
        self.only_overwrite_undefined = False
        self.min_bout_length = 5

        # initialize model training variables
        self.training_epochs_per_img = 9
        self.delete_training_images = True
        self.training_image_dir = ""
        self.model_type = DEFAULT_MODEL_TYPE

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
        self.ui.select_calibration_button.clicked.connect(self.select_calibration_file)
        self.ui.load_model_button.clicked.connect(self.load_model)
        self.ui.score_all_button.clicked.connect(self.score_all)
        self.ui.overwritecheckbox.stateChanged.connect(self.update_overwrite_policy)
        self.ui.bout_length_input.valueChanged.connect(self.update_min_bout_length)
        self.ui.user_manual_button.clicked.connect(self.show_user_manual)
        self.ui.image_number_input.valueChanged.connect(self.update_epochs_per_img)
        self.ui.delete_image_box.stateChanged.connect(self.update_image_deletion)
        self.ui.training_folder_button.clicked.connect(self.set_training_folder)
        self.ui.train_model_button.clicked.connect(self.train_model)
        self.ui.save_config_button.clicked.connect(self.save_brain_state_config)

        # user input: drag and drop
        self.ui.recording_file_label.installEventFilter(self)
        self.ui.label_file_label.installEventFilter(self)
        self.ui.calibration_file_label.installEventFilter(self)
        self.ui.model_label.installEventFilter(self)

        self.show()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Filter mouse events to detect when user drags/drops a file

        :param obj: UI object receiving the event
        :param event: mouse event
        :return: whether to filter (block) the event
        """
        filename = None
        if obj in [
            self.ui.recording_file_label,
            self.ui.label_file_label,
            self.ui.calibration_file_label,
            self.ui.model_label,
        ]:
            event.accept()
            if event.type() == QtCore.QEvent.Drop:
                urls = event.mimeData().urls()
                if len(urls) == 1:
                    filename = urls[0].toLocalFile()

        if filename is None:
            return super().eventFilter(obj, event)

        _, file_extension = os.path.splitext(filename)

        if obj == self.ui.recording_file_label:
            if file_extension in RECORDING_FILE_TYPES:
                self.recordings[self.recording_index].recording_file = filename
                self.ui.recording_file_label.setText(filename)
        elif obj == self.ui.label_file_label:
            if file_extension == LABEL_FILE_TYPE:
                self.recordings[self.recording_index].label_file = filename
                self.ui.label_file_label.setText(filename)
        elif obj == self.ui.calibration_file_label:
            if file_extension == CALIBRATION_FILE_TYPE:
                self.recordings[self.recording_index].calibration_file = filename
                self.ui.calibration_file_label.setText(filename)
        elif obj == self.ui.model_label:
            try:
                self.model = load_model(
                    filename=filename, n_classes=self.brain_state_set.n_classes
                )
            except Exception:
                self.show_message(f"ERROR: could not load model from {filename} ")
                return super().eventFilter(obj, event)
            self.ui.model_label.setText(filename)

        return super().eventFilter(obj, event)

    def train_model(self) -> None:
        # check basic training inputs
        if (
            self.model_type == DEFAULT_MODEL_TYPE
            and self.training_epochs_per_img % 2 == 0
        ):
            self.show_message(
                (
                    "ERROR: for the default model type, number of epochs "
                    "per image must be an odd number."
                )
            )
            return
        if self.training_image_dir == "":
            self.show_message("ERROR: no folder selected for training images.")
            return

        # check some inputs for each recording
        for recording_index in range(len(self.recordings)):
            error_message = self.check_single_file_inputs(recording_index)
            if error_message:
                self.show_message(
                    f"ERROR ({self.recordings[recording_index].name}): {error_message}"
                )
                return

        # get filename for the new model
        model_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Save classification model file as",
            filter="*" + MODEL_FILE_TYPE,
        )
        if not model_filename:
            self.show_message("Model training canceled, no filename given")

        # create image folder
        if os.path.exists(self.training_image_dir):
            self.show_message(
                f"Warning: training image folder exists, will be overwritten"
            )
        os.makedirs(self.training_image_dir, exist_ok=True)

        # create training images
        self.show_message(
            (
                f"Creating training images in {self.training_image_dir}, "
                "please wait..."
            )
        )
        self.ui.message_area.repaint()
        app.processEvents()
        failed_recordings = create_training_images(
            recordings=self.recordings,
            output_path=self.training_image_dir,
            epoch_length=self.epoch_length,
            epochs_per_img=self.training_epochs_per_img,
            brain_state_set=self.brain_state_set,
        )
        if len(failed_recordings) > 0:
            if len(failed_recordings) == len(self.recordings):
                self.show_message(f"ERROR: no recordings were valid!")
            else:
                self.show_message(
                    (
                        "WARNING: the following recordings could not be"
                        "loaded and will not be used for training: "
                        f"{', '.join([str(r) for r in failed_recordings])}"
                    )
                )

        # train model
        self.show_message(f"Training model, please wait...")
        self.ui.message_area.repaint()
        app.processEvents()
        model = train_model(
            annotations_file=os.path.join(
                self.training_image_dir, ANNOTATIONS_FILENAME
            ),
            img_dir=self.training_image_dir,
            epochs_per_image=self.training_epochs_per_img,
            model_type=self.model_type,
            mixture_weights=self.brain_state_set.mixture_weights,
            n_classes=self.brain_state_set.n_classes,
        )

        # save model
        save_model(model=model, filename=model_filename)

        # optionally delete images
        if self.delete_training_images:
            shutil.rmtree(self.training_image_dir)

        self.show_message(f"Training complete, saved model to {model_filename}")

    def set_training_folder(self):
        training_folder_parent = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select directory for training images"
        )
        if training_folder_parent:
            self.training_image_dir = os.path.join(
                training_folder_parent,
                "images_" + datetime.datetime.now().strftime("%Y%m%d%H%M"),
            )
            self.ui.image_folder_label.setText(self.training_image_dir)

    def update_image_deletion(self) -> None:
        """Update choice of whether to delete images after training"""
        self.delete_training_images = self.ui.delete_image_box.isChecked()

    def update_epochs_per_img(self, new_value) -> None:
        """Update number of epochs per image

        :param new_value: new number of epochs per image
        """
        self.training_epochs_per_img = new_value

    def score_all(self) -> None:
        """Score all recordings using the classification model"""
        # check basic inputs
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
            if self.recordings[recording_index].calibration_file == "":
                self.ui.score_all_status.setText(
                    f"error on recording {self.recordings[recording_index].name}"
                )
                self.show_message(
                    (
                        f"ERROR ({self.recordings[recording_index].name}): "
                        "no calibration file selected"
                    )
                )
                return

        # score each recording
        for recording_index in range(len(self.recordings)):
            # load EEG, EMG
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

            # load labels
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
                epochs_in_recording = round(eeg.size / samples_per_epoch)
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

            # load calibration data
            if not os.path.isfile(self.recordings[recording_index].calibration_file):
                self.show_message(
                    (
                        "ERROR: calibration file does not exist for recording "
                        f"{self.recordings[recording_index].name}. "
                        "This recording will be skipped."
                    )
                )
                continue
            try:
                (
                    mixture_means,
                    mixture_sds,
                ) = load_calibration_file(
                    self.recordings[recording_index].calibration_file
                )
            except Exception:
                self.show_message(
                    (
                        "ERROR: could not load calibration file for recording "
                        f"{self.recordings[recording_index].name}. "
                        "This recording will be skipped."
                    )
                )
                continue

            labels = score_recording(
                model=self.model,
                eeg=eeg,
                emg=emg,
                mixture_means=mixture_means,
                mixture_sds=mixture_sds,
                sampling_rate=sampling_rate,
                epoch_length=self.epoch_length,
                brain_state_set=self.brain_state_set,
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
        file_dialog.setNameFilter("*" + MODEL_FILE_TYPE)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            if not os.path.isfile(filename):
                self.show_message("ERROR: model file does not exist")
                return
            try:
                self.model = load_model(
                    filename=filename, n_classes=self.brain_state_set.n_classes
                )
            except Exception:
                self.show_message(
                    (
                        "ERROR: could not load classification model. Check "
                        "user manual for instructions on creating this file."
                    )
                )
                return

            self.ui.model_label.setText(filename)

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
        all valid, creates the calibration file, and sets the
        "calibration file" property of the current recording to be the
        newly created file.
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
            brain_state_set=self.brain_state_set,
        )
        if label_error_message:
            self.ui.calibration_status.setText("invalid label file")
            self.show_message(f"ERROR: {label_error_message}")
            return

        # get the name for the calibration file
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Save calibration file as",
            filter="*" + CALIBRATION_FILE_TYPE,
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
            brain_state_set=self.brain_state_set,
        )

        self.ui.calibration_status.setText("")
        self.show_message(
            (
                "Created calibration file using recording "
                f"{self.recordings[self.recording_index].name} "
                f"at {filename}"
            )
        )

        self.recordings[self.recording_index].calibration_file = filename
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
            self.ui.manual_scoring_status
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
            brain_state_set=self.brain_state_set,
        )
        if label_error:
            # if the label length is only off by one, pad or truncate as needed
            # and show a warning
            if label_error == LABEL_LENGTH_ERROR:
                # should be very close to an integer
                samples_per_epoch = round(sampling_rate * self.epoch_length)
                epochs_in_recording = round(eeg.size / samples_per_epoch)
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
            filter="*" + LABEL_FILE_TYPE,
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
        file_dialog.setNameFilter("*" + LABEL_FILE_TYPE)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            self.recordings[self.recording_index].label_file = filename
            self.ui.label_file_label.setText(filename)

    def select_calibration_file(self) -> None:
        """User can select a calibration file"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select calibration file")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("*" + CALIBRATION_FILE_TYPE)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            self.recordings[self.recording_index].calibration_file = filename
            self.ui.calibration_file_label.setText(filename)

    def select_recording_file(self) -> None:
        """User can select a recording file"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select recording file")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter(f"(*{' *'.join(RECORDING_FILE_TYPES)})")

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
        self.ui.calibration_file_label.setText(
            self.recordings[self.recording_index].calibration_file
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
            f"Data / actions for Recording {self.recordings[list_index].name}"
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
            self.recording_index = self.ui.recording_list_widget.currentRow()

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
        scroll_area.setStyleSheet("background-color: white;")
        scroll_area.setWidget(label_widget)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(scroll_area)
        self.popup = QtWidgets.QWidget()
        self.popup.setLayout(grid)
        self.popup.setGeometry(QtCore.QRect(100, 100, 600, 600))
        self.popup.show()

    def initialize_settings_tab(self):
        """Populate settings tab and assign its callbacks"""
        # show information about the settings tab
        config_guide_file = open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_GUIDE_FILE),
            "r",
        )
        config_guide_text = config_guide_file.read()
        config_guide_file.close()
        self.ui.settings_text.setText(config_guide_text)

        # store dictionary that maps digits to rows of widgets
        # in the settings tab
        self.settings_widgets = {
            1: StateSettings(
                digit=1,
                enabled_widget=self.ui.enable_state_1,
                name_widget=self.ui.state_name_1,
                is_scored_widget=self.ui.state_scored_1,
                frequency_widget=self.ui.state_frequency_1,
            ),
            2: StateSettings(
                digit=2,
                enabled_widget=self.ui.enable_state_2,
                name_widget=self.ui.state_name_2,
                is_scored_widget=self.ui.state_scored_2,
                frequency_widget=self.ui.state_frequency_2,
            ),
            3: StateSettings(
                digit=3,
                enabled_widget=self.ui.enable_state_3,
                name_widget=self.ui.state_name_3,
                is_scored_widget=self.ui.state_scored_3,
                frequency_widget=self.ui.state_frequency_3,
            ),
            4: StateSettings(
                digit=4,
                enabled_widget=self.ui.enable_state_4,
                name_widget=self.ui.state_name_4,
                is_scored_widget=self.ui.state_scored_4,
                frequency_widget=self.ui.state_frequency_4,
            ),
            5: StateSettings(
                digit=5,
                enabled_widget=self.ui.enable_state_5,
                name_widget=self.ui.state_name_5,
                is_scored_widget=self.ui.state_scored_5,
                frequency_widget=self.ui.state_frequency_5,
            ),
            6: StateSettings(
                digit=6,
                enabled_widget=self.ui.enable_state_6,
                name_widget=self.ui.state_name_6,
                is_scored_widget=self.ui.state_scored_6,
                frequency_widget=self.ui.state_frequency_6,
            ),
            7: StateSettings(
                digit=7,
                enabled_widget=self.ui.enable_state_7,
                name_widget=self.ui.state_name_7,
                is_scored_widget=self.ui.state_scored_7,
                frequency_widget=self.ui.state_frequency_7,
            ),
            8: StateSettings(
                digit=8,
                enabled_widget=self.ui.enable_state_8,
                name_widget=self.ui.state_name_8,
                is_scored_widget=self.ui.state_scored_8,
                frequency_widget=self.ui.state_frequency_8,
            ),
            9: StateSettings(
                digit=9,
                enabled_widget=self.ui.enable_state_9,
                name_widget=self.ui.state_name_9,
                is_scored_widget=self.ui.state_scored_9,
                frequency_widget=self.ui.state_frequency_9,
            ),
            0: StateSettings(
                digit=0,
                enabled_widget=self.ui.enable_state_0,
                name_widget=self.ui.state_name_0,
                is_scored_widget=self.ui.state_scored_0,
                frequency_widget=self.ui.state_frequency_0,
            ),
        }

        # update widget state to display current config
        states = {b.digit: b for b in self.brain_state_set.brain_states}
        for digit in range(10):
            if digit in states.keys():
                self.settings_widgets[digit].enabled_widget.setChecked(True)
                self.settings_widgets[digit].name_widget.setText(states[digit].name)
                self.settings_widgets[digit].is_scored_widget.setChecked(
                    states[digit].is_scored
                )
                self.settings_widgets[digit].frequency_widget.setValue(
                    states[digit].frequency
                )
            else:
                self.settings_widgets[digit].enabled_widget.setChecked(False)
                self.settings_widgets[digit].name_widget.setEnabled(False)
                self.settings_widgets[digit].is_scored_widget.setEnabled(False)
                self.settings_widgets[digit].frequency_widget.setEnabled(False)

        # set callbacks
        for digit in range(10):
            state = self.settings_widgets[digit]
            state.enabled_widget.stateChanged.connect(
                partial(self.set_brain_state_enabled, digit)
            )
            state.name_widget.editingFinished.connect(self.finished_editing_state_name)
            state.is_scored_widget.stateChanged.connect(
                partial(self.is_scored_changed, digit)
            )
            state.frequency_widget.valueChanged.connect(self.state_frequency_changed)

    def set_brain_state_enabled(self, digit, e) -> None:
        """Called when user clicks "enabled" checkbox

        :param digit: brain state digit
        :param e: unused but mandatory
        """
        # get the widgets for this brain state
        state = self.settings_widgets[digit]
        # update state of these widgets
        is_checked = state.enabled_widget.isChecked()
        for widget in [
            state.name_widget,
            state.is_scored_widget,
        ]:
            widget.setEnabled(is_checked)
        state.frequency_widget.setEnabled(
            is_checked and state.is_scored_widget.isChecked()
        )
        if not is_checked:
            state.name_widget.setText("")
            state.frequency_widget.setValue(0)
        # check that configuration is valid
        _ = self.check_config_validity()

    def finished_editing_state_name(self) -> None:
        """Called when user finishes editing a brain state's name"""
        _ = self.check_config_validity()

    def state_frequency_changed(self, new_value) -> None:
        """Called when user edits a brain state's frequency

        :param new_value: unused
        """
        _ = self.check_config_validity()

    def is_scored_changed(self, digit, e) -> None:
        """Called when user sets whether a state is scored

        :param digit: brain state digit
        :param e: unused, but mandatory
        """
        # get the widgets for this brain state
        state = self.settings_widgets[digit]
        # update the state of these widgets
        is_checked = state.is_scored_widget.isChecked()
        state.frequency_widget.setEnabled(is_checked)
        if not is_checked:
            state.frequency_widget.setValue(0)
        # check that configuration is valid
        _ = self.check_config_validity()

    def check_config_validity(self) -> str:
        """Check if brain state configuration on screen is valid"""
        # error message, if we get one
        message = None

        # strip whitespace from brain state names and update display
        for digit in range(10):
            state = self.settings_widgets[digit]
            current_name = state.name_widget.text()
            formatted_name = current_name.strip()
            if current_name != formatted_name:
                state.name_widget.setText(formatted_name)

        # check if names are unique and frequencies add up to 1
        names = []
        frequencies = []
        for digit in range(10):
            state = self.settings_widgets[digit]
            if state.enabled_widget.isChecked():
                names.append(state.name_widget.text())
                frequencies.append(state.frequency_widget.value())
        if len(names) != len(set(names)):
            message = "Error: names must be unique"
        if sum(frequencies) != 1:
            message = "Error: sum(frequencies) != 1"

        if message is not None:
            self.ui.save_config_status.setText(message)
            self.ui.save_config_button.setEnabled(False)
            return message

        self.ui.save_config_button.setEnabled(True)
        self.ui.save_config_status.setText("")

    def save_brain_state_config(self):
        """Save configuration to file"""
        # check that configuration is valid
        error_message = self.check_config_validity()
        if error_message is not None:
            return

        # build a BrainStateMapper object from the current configuration
        brain_states = list()
        for digit in range(10):
            state = self.settings_widgets[digit]
            if state.enabled_widget.isChecked():
                brain_states.append(
                    BrainState(
                        name=state.name_widget.text(),
                        digit=digit,
                        is_scored=state.is_scored_widget.isChecked(),
                        frequency=state.frequency_widget.value(),
                    )
                )
        self.brain_state_set = BrainStateSet(brain_states, UNDEFINED_LABEL)

        # save to file
        save_config(self.brain_state_set)
        self.ui.save_config_status.setText("configuration saved")


def check_label_validity(
    labels: np.array,
    samples_in_recording: int,
    sampling_rate: int | float,
    epoch_length: int | float,
    brain_state_set: BrainStateSet,
) -> str:
    """Check whether a set of brain state labels is valid

    This returns an error message if a problem is found with the
    brain state labels.

    :param labels: brain state labels
    :param samples_in_recording: number of samples in the recording
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :param brain_state_set: BrainStateMapper object
    :return: error message
    """
    # check that length is correct
    samples_per_epoch = round(sampling_rate * epoch_length)
    epochs_in_recording = round(samples_in_recording / samples_per_epoch)
    if epochs_in_recording != labels.size:
        return LABEL_LENGTH_ERROR

    # check that entries are valid
    if not set(labels.tolist()).issubset(
        set([b.digit for b in brain_state_set.brain_states] + [UNDEFINED_LABEL])
    ):
        return "label file contains invalid entries"


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AccuSleepWindow()
    sys.exit(app.exec())
