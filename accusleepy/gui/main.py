# AccuSleePy main window
# Icon source: Arkinasi, https://www.flaticon.com/authors/arkinasi

import datetime
import os
import shutil
import sys
from dataclasses import dataclass
from functools import partial

import numpy as np
import toml
from PySide6.QtCore import (
    QEvent,
    QKeyCombination,
    QObject,
    QRect,
    Qt,
    QUrl,
)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from accusleepy.bouts import enforce_min_bout_length
from accusleepy.brain_state_set import BRAIN_STATES_KEY, BrainState, BrainStateSet
from accusleepy.constants import (
    ANNOTATIONS_FILENAME,
    CALIBRATION_ANNOTATION_FILENAME,
    CALIBRATION_FILE_TYPE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_EMG_FILTER_ORDER,
    DEFAULT_EMG_BP_LOWER,
    DEFAULT_EMG_BP_UPPER,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOMENTUM,
    DEFAULT_TRAINING_EPOCHS,
    LABEL_FILE_TYPE,
    MESSAGE_BOX_MAX_DEPTH,
    MODEL_FILE_TYPE,
    REAL_TIME_MODEL_TYPE,
    RECORDING_FILE_TYPES,
    RECORDING_LIST_FILE_TYPE,
    UNDEFINED_LABEL,
)
from accusleepy.fileio import (
    Recording,
    load_calibration_file,
    load_config,
    load_labels,
    load_recording,
    load_recording_list,
    save_config,
    save_labels,
    save_recording_list,
    EMGFilter,
    Hyperparameters,
)
from accusleepy.gui.manual_scoring import ManualScoringWindow
from accusleepy.gui.primary_window import Ui_PrimaryWindow
from accusleepy.signal_processing import (
    create_training_images,
    resample_and_standardize,
)
from accusleepy.validation import (
    check_label_validity,
    LABEL_LENGTH_ERROR,
    check_config_consistency,
)

# note: functions using torch or scipy are lazily imported

# on Windows, prevent dark mode from changing the visual style
sys.argv += ["-platform", "windows:darkmode=1"]


# relative path to user manual
MAIN_GUIDE_FILE = os.path.normpath(r"text/main_guide.md")


@dataclass
class StateSettings:
    """Widgets for config settings for a brain state"""

    digit: int
    enabled_widget: QCheckBox
    name_widget: QLabel
    is_scored_widget: QCheckBox
    frequency_widget: QDoubleSpinBox


class AccuSleepWindow(QMainWindow):
    """AccuSleePy primary window"""

    def __init__(self):
        super(AccuSleepWindow, self).__init__()

        # initialize the UI
        self.ui = Ui_PrimaryWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy")

        # fill in settings tab
        (
            self.brain_state_set,
            self.epoch_length,
            self.only_overwrite_undefined,
            self.save_confidence_scores,
            self.min_bout_length,
            self.emg_filter,
            self.hyperparameters,
        ) = load_config()

        self.settings_widgets = None
        self.initialize_settings_tab()

        # initialize info about the recordings, classification data / settings
        self.ui.epoch_length_input.setValue(self.epoch_length)
        self.ui.overwritecheckbox.setChecked(self.only_overwrite_undefined)
        self.ui.save_confidence_checkbox.setChecked(self.save_confidence_scores)
        self.ui.bout_length_input.setValue(self.min_bout_length)
        self.model = None

        # initialize model training variables
        self.training_epochs_per_img = 9
        self.delete_training_images = True
        self.model_type = DEFAULT_MODEL_TYPE
        self.calibrate_trained_model = True

        # metadata for the currently loaded classification model
        self.model_epoch_length = None
        self.model_epochs_per_img = None

        # set up the list of recordings
        first_recording = Recording(
            widget=QListWidgetItem("Recording 1", self.ui.recording_list_widget),
        )
        self.ui.recording_list_widget.addItem(first_recording.widget)
        self.ui.recording_list_widget.setCurrentRow(0)
        # index of currently selected recording in the list
        self.recording_index = 0
        # list of recordings the user has added
        self.recordings = [first_recording]

        # messages to display
        self.messages = []

        # display current version
        version = ""
        toml_file = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "pyproject.toml",
        )
        if os.path.isfile(toml_file):
            toml_data = toml.load(toml_file)
            if "project" in toml_data and "version" in toml_data["project"]:
                version = toml_data["project"]["version"]
        self.ui.version_label.setText(f"v{version}")

        # user input: keyboard shortcuts
        keypress_quit = QShortcut(
            QKeySequence(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_W)),
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
        self.ui.load_model_button.clicked.connect(partial(self.load_model, None))
        self.ui.score_all_button.clicked.connect(self.score_all)
        self.ui.overwritecheckbox.stateChanged.connect(self.update_overwrite_policy)
        self.ui.save_confidence_checkbox.stateChanged.connect(
            self.update_confidence_policy
        )
        self.ui.bout_length_input.valueChanged.connect(self.update_min_bout_length)
        self.ui.user_manual_button.clicked.connect(self.show_user_manual)
        self.ui.image_number_input.valueChanged.connect(self.update_epochs_per_img)
        self.ui.delete_image_box.stateChanged.connect(self.update_image_deletion)
        self.ui.calibrate_checkbox.stateChanged.connect(
            self.update_training_calibration
        )
        self.ui.train_model_button.clicked.connect(self.train_model)
        self.ui.save_config_button.clicked.connect(self.save_brain_state_config)
        self.ui.export_button.clicked.connect(self.export_recording_list)
        self.ui.import_button.clicked.connect(self.import_recording_list)
        self.ui.default_type_button.toggled.connect(self.model_type_radio_buttons)
        self.ui.reset_emg_params_button.clicked.connect(self.reset_emg_filter_settings)
        self.ui.reset_hyperparams_button.clicked.connect(
            self.reset_hyperparams_settings
        )

        # user input: drag and drop
        self.ui.recording_file_label.installEventFilter(self)
        self.ui.label_file_label.installEventFilter(self)
        self.ui.calibration_file_label.installEventFilter(self)
        self.ui.model_label.installEventFilter(self)

        self.show()

    def model_type_radio_buttons(self, default_selected: bool) -> None:
        """Toggle training default or real-time model

        :param default_selected: whether default option is selected
        """
        self.model_type = (
            DEFAULT_MODEL_TYPE if default_selected else REAL_TIME_MODEL_TYPE
        )

    def export_recording_list(self) -> None:
        """Save current list of recordings to file"""
        # get the name for the recording list file
        filename, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save list of recordings as",
            filter="*" + RECORDING_LIST_FILE_TYPE,
        )
        if not filename:
            return
        filename = os.path.normpath(filename)
        save_recording_list(filename=filename, recordings=self.recordings)
        self.show_message(f"Saved list of recordings to {filename}")

    def import_recording_list(self):
        """Load list of recordings from file, overwriting current list"""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select list of recordings")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("*" + RECORDING_LIST_FILE_TYPE)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            filename = os.path.normpath(filename)
        else:
            return

        # clear widget
        self.ui.recording_list_widget.clear()
        # overwrite current list
        self.recordings = load_recording_list(filename)

        for recording in self.recordings:
            recording.widget = QListWidgetItem(
                f"Recording {recording.name}", self.ui.recording_list_widget
            )
            self.ui.recording_list_widget.addItem(self.recordings[-1].widget)

        # display new list
        self.ui.recording_list_widget.setCurrentRow(0)
        self.show_message(f"Loaded list of recordings from {filename}")

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
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
            if event.type() == QEvent.Drop:
                urls = event.mimeData().urls()
                if len(urls) == 1:
                    filename = os.path.normpath(urls[0].toLocalFile())

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
            self.load_model(filename=filename)

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

        # determine fraction of training data to use for calibration
        if self.calibrate_trained_model:
            calibration_fraction = self.ui.calibration_spinbox.value() / 100
        else:
            calibration_fraction = 0

        # check some inputs for each recording
        for recording_index in range(len(self.recordings)):
            error_message = self.check_single_file_inputs(recording_index)
            if error_message:
                self.show_message(
                    f"ERROR (recording {self.recordings[recording_index].name}): {error_message}"
                )
                return

        # get filename for the new model
        model_filename, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save classification model file as",
            filter="*" + MODEL_FILE_TYPE,
        )
        if not model_filename:
            self.show_message("Model training canceled, no filename given")
            return
        model_filename = os.path.normpath(model_filename)

        # create (probably temporary) image folder in
        # the same folder as the trained model
        temp_image_dir = os.path.join(
            os.path.dirname(model_filename),
            "images_" + datetime.datetime.now().strftime("%Y%m%d%H%M"),
        )

        if os.path.exists(temp_image_dir):  # unlikely
            self.show_message(
                "Warning: training image folder exists, will be overwritten"
            )
        os.makedirs(temp_image_dir, exist_ok=True)

        # create training images
        self.show_message("Training, please wait. See console for progress updates.")
        if not self.delete_training_images:
            self.show_message((f"Creating training images in {temp_image_dir}"))
        else:
            self.show_message(
                (f"Creating temporary folder of training images: {temp_image_dir}")
            )
        self.ui.message_area.repaint()
        QApplication.processEvents()
        print("Creating training images")
        failed_recordings = create_training_images(
            recordings=self.recordings,
            output_path=temp_image_dir,
            epoch_length=self.epoch_length,
            epochs_per_img=self.training_epochs_per_img,
            brain_state_set=self.brain_state_set,
            model_type=self.model_type,
            calibration_fraction=calibration_fraction,
            emg_filter=self.emg_filter,
        )
        if len(failed_recordings) > 0:
            if len(failed_recordings) == len(self.recordings):
                self.show_message("ERROR: no recordings were valid!")
                return
            else:
                self.show_message(
                    (
                        "WARNING: the following recordings could not be "
                        "loaded and will not be used for training: "
                        f"{', '.join([str(r) for r in failed_recordings])}"
                    )
                )

        # train model
        self.show_message("Training model")
        self.ui.message_area.repaint()
        QApplication.processEvents()
        print("Training model")
        from accusleepy.classification import create_dataloader, train_ssann
        from accusleepy.models import save_model
        from accusleepy.temperature_scaling import ModelWithTemperature

        model = train_ssann(
            annotations_file=os.path.join(temp_image_dir, ANNOTATIONS_FILENAME),
            img_dir=temp_image_dir,
            mixture_weights=self.brain_state_set.mixture_weights,
            n_classes=self.brain_state_set.n_classes,
            hyperparameters=self.hyperparameters,
        )

        # calibrate the model
        if self.calibrate_trained_model:
            calibration_annotation_file = os.path.join(
                temp_image_dir, CALIBRATION_ANNOTATION_FILENAME
            )
            calibration_dataloader = create_dataloader(
                annotations_file=calibration_annotation_file,
                img_dir=temp_image_dir,
                hyperparameters=self.hyperparameters,
            )
            model = ModelWithTemperature(model)
            print("Calibrating model")
            model.set_temperature(calibration_dataloader)

        # save model
        save_model(
            model=model,
            filename=model_filename,
            epoch_length=self.epoch_length,
            epochs_per_img=self.training_epochs_per_img,
            model_type=self.model_type,
            brain_state_set=self.brain_state_set,
            is_calibrated=self.calibrate_trained_model,
        )

        # optionally delete images
        if self.delete_training_images:
            print("Cleaning up training image folder")
            shutil.rmtree(temp_image_dir)

        self.show_message(f"Training complete. Saved model to {model_filename}")
        print("Training complete.")

    def update_image_deletion(self) -> None:
        """Update choice of whether to delete images after training"""
        self.delete_training_images = self.ui.delete_image_box.isChecked()

    def update_training_calibration(self) -> None:
        """Update choice of whether to calibrate model after training"""
        self.calibrate_trained_model = self.ui.calibrate_checkbox.isChecked()
        self.ui.calibration_spinbox.setEnabled(self.calibrate_trained_model)

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
        if self.epoch_length != self.model_epoch_length:
            self.ui.score_all_status.setText("invalid epoch length")
            self.show_message(
                (
                    "ERROR: model was trained with an epoch length of "
                    f"{self.model_epoch_length} seconds, but the current "
                    f"epoch length setting is {self.epoch_length} seconds."
                )
            )
            return

        self.ui.score_all_status.setText("running...")
        self.ui.score_all_status.repaint()
        QApplication.processEvents()

        from accusleepy.classification import score_recording

        # check some inputs for each recording
        for recording_index in range(len(self.recordings)):
            error_message = self.check_single_file_inputs(recording_index)
            if error_message:
                self.ui.score_all_status.setText(
                    f"error on recording {self.recordings[recording_index].name}"
                )
                self.show_message(
                    f"ERROR (recording {self.recordings[recording_index].name}): {error_message}"
                )
                return
            if self.recordings[recording_index].calibration_file == "":
                self.ui.score_all_status.setText(
                    f"error on recording {self.recordings[recording_index].name}"
                )
                self.show_message(
                    (
                        f"ERROR (recording {self.recordings[recording_index].name}): "
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
                    # ignore any existing confidence scores; they will all be overwritten
                    existing_labels, _ = load_labels(label_file)
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

            labels, confidence_scores = score_recording(
                model=self.model,
                eeg=eeg,
                emg=emg,
                mixture_means=mixture_means,
                mixture_sds=mixture_sds,
                sampling_rate=sampling_rate,
                epoch_length=self.epoch_length,
                epochs_per_img=self.model_epochs_per_img,
                brain_state_set=self.brain_state_set,
                emg_filter=self.emg_filter,
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

            # ignore confidence scores if desired
            if not self.save_confidence_scores:
                confidence_scores = None

            # save results
            save_labels(
                labels=labels, filename=label_file, confidence_scores=confidence_scores
            )
            self.show_message(
                (
                    "Saved labels for recording "
                    f"{self.recordings[recording_index].name} "
                    f"to {label_file}"
                )
            )

        self.ui.score_all_status.setText("")

    def load_model(self, filename=None) -> None:
        """Load trained classification model from file

        :param filename: model filename, if it's known
        """
        if filename is None:
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Select classification model")
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
            file_dialog.setNameFilter("*" + MODEL_FILE_TYPE)

            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                filename = selected_files[0]
                filename = os.path.normpath(filename)
            else:
                return

        if not os.path.isfile(filename):
            self.show_message("ERROR: model file does not exist")
            return

        self.show_message("Loading classification model")
        self.ui.message_area.repaint()
        QApplication.processEvents()

        from accusleepy.models import load_model

        try:
            model, epoch_length, epochs_per_img, model_type, brain_states = load_model(
                filename=filename
            )
        except Exception:
            self.show_message(
                (
                    "ERROR: could not load classification model. Check "
                    "user manual for instructions on creating this file."
                )
            )
            return

        # make sure only "default" model type is loaded
        if model_type != DEFAULT_MODEL_TYPE:
            self.show_message(
                (
                    "ERROR: only 'default'-style models can be used. "
                    "'Real-time' models are not supported. "
                    "See classification.example_real_time_scoring_function.py "
                    "for an example of how to classify brain states in real time."
                )
            )
            return

        self.model = model
        self.model_epoch_length = epoch_length
        self.model_epochs_per_img = epochs_per_img

        # warn user if the model's expected epoch length or brain states
        # don't match the current configuration
        config_warnings = check_config_consistency(
            current_brain_states=self.brain_state_set.to_output_dict()[
                BRAIN_STATES_KEY
            ],
            model_brain_states=brain_states,
            current_epoch_length=self.epoch_length,
            model_epoch_length=epoch_length,
        )
        if len(config_warnings) > 0:
            for w in config_warnings:
                self.show_message(w)
        else:
            self.show_message(f"Loaded classification model from {filename}")

        self.ui.model_label.setText(filename)

    def load_single_recording(
        self, status_widget: QLabel
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
            labels, _ = load_labels(label_file)
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
            confidence_scores=None,
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
        filename, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save calibration file as",
            filter="*" + CALIBRATION_FILE_TYPE,
        )
        if not filename:
            return
        filename = os.path.normpath(filename)

        from accusleepy.classification import create_calibration_file

        create_calibration_file(
            filename=filename,
            eeg=eeg,
            emg=emg,
            labels=labels,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
            brain_state_set=self.brain_state_set,
            emg_filter=self.emg_filter,
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

    def check_single_file_inputs(self, recording_index: int) -> str | None:
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
        if not os.path.isfile(self.recordings[self.recording_index].recording_file):
            return "recording file does not exist"
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

    def update_confidence_policy(self, checked) -> None:
        """Toggle policy for saving confidence scores

        If the checkbox is enabled, confidence scores will be saved to the label files.

        :param checked: state of the checkbox
        """
        self.save_confidence_scores = checked

    def manual_scoring(self) -> None:
        """View the selected recording for manual scoring"""
        # immediately display a status message
        self.ui.manual_scoring_status.setText("loading...")
        self.ui.manual_scoring_status.repaint()
        QApplication.processEvents()

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
                labels, confidence_scores = load_labels(label_file)
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
            # manual scoring will not add a new confidence score column
            # to a label file that does not have one
            confidence_scores = None

        # check that all labels are valid
        label_error = check_label_validity(
            labels=labels,
            confidence_scores=confidence_scores,
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
                    if confidence_scores is not None:
                        confidence_scores = np.concatenate(
                            (confidence_scores, np.array([0]))
                        )
                    self.show_message(
                        (
                            "WARNING: an undefined epoch was added to "
                            "the label file to correct its length."
                        )
                    )
                elif labels.size - epochs_in_recording == 1:
                    labels = labels[:-1]
                    if confidence_scores is not None:
                        confidence_scores = confidence_scores[:-1]
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
            confidence_scores=confidence_scores,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
            emg_filter=self.emg_filter,
        )
        manual_scoring_window.setWindowTitle(f"AccuSleePy viewer: {label_file}")
        manual_scoring_window.exec()
        self.ui.manual_scoring_status.setText("")

    def create_label_file(self) -> None:
        """Set the filename for a new label file"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            caption="Set filename for label file (nothing will be overwritten yet)",
            filter="*" + LABEL_FILE_TYPE,
        )
        if filename:
            filename = os.path.normpath(filename)
            self.recordings[self.recording_index].label_file = filename
            self.ui.label_file_label.setText(filename)

    def select_label_file(self) -> None:
        """User can select an existing label file"""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select label file")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("*" + LABEL_FILE_TYPE)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            filename = os.path.normpath(filename)
            self.recordings[self.recording_index].label_file = filename
            self.ui.label_file_label.setText(filename)

    def select_calibration_file(self) -> None:
        """User can select a calibration file"""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select calibration file")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("*" + CALIBRATION_FILE_TYPE)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            filename = os.path.normpath(filename)
            self.recordings[self.recording_index].calibration_file = filename
            self.ui.calibration_file_label.setText(filename)

    def select_recording_file(self) -> None:
        """User can select a recording file"""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select recording file")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter(f"(*{' *'.join(RECORDING_FILE_TYPES)})")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            filename = os.path.normpath(filename)
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
                widget=QListWidgetItem(
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
        self.popup = QWidget()
        self.popup_vlayout = QVBoxLayout(self.popup)
        self.guide_textbox = QTextBrowser(self.popup)
        self.popup_vlayout.addWidget(self.guide_textbox)

        url = QUrl.fromLocalFile(MAIN_GUIDE_FILE)
        self.guide_textbox.setSource(url)
        self.guide_textbox.setOpenLinks(False)

        self.popup.setGeometry(QRect(100, 100, 600, 600))
        self.popup.show()

    def initialize_settings_tab(self):
        """Populate settings tab and assign its callbacks"""
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
        # UI defaults
        self.ui.default_epoch_input.setValue(self.epoch_length)
        self.ui.overwrite_default_checkbox.setChecked(self.only_overwrite_undefined)
        self.ui.confidence_setting_checkbox.setChecked(self.save_confidence_scores)
        self.ui.default_min_bout_length_spinbox.setValue(self.min_bout_length)
        # EMG filter
        self.ui.emg_order_spinbox.setValue(self.emg_filter.order)
        self.ui.bp_lower_spinbox.setValue(self.emg_filter.bp_lower)
        self.ui.bp_upper_spinbox.setValue(self.emg_filter.bp_upper)
        # model training hyperparameters
        self.ui.batch_size_spinbox.setValue(self.hyperparameters.batch_size)
        self.ui.learning_rate_spinbox.setValue(self.hyperparameters.learning_rate)
        self.ui.momentum_spinbox.setValue(self.hyperparameters.momentum)
        self.ui.training_epochs_spinbox.setValue(self.hyperparameters.training_epochs)
        # brain states
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
        self.ui.emg_order_spinbox.valueChanged.connect(self.emg_filter_order_changed)
        self.ui.bp_lower_spinbox.valueChanged.connect(self.emg_filter_bp_lower_changed)
        self.ui.bp_upper_spinbox.valueChanged.connect(self.emg_filter_bp_upper_changed)
        self.ui.batch_size_spinbox.valueChanged.connect(self.hyperparameters_changed)
        self.ui.learning_rate_spinbox.valueChanged.connect(self.hyperparameters_changed)
        self.ui.momentum_spinbox.valueChanged.connect(self.hyperparameters_changed)
        self.ui.training_epochs_spinbox.valueChanged.connect(
            self.hyperparameters_changed
        )
        for digit in range(10):
            state = self.settings_widgets[digit]
            state.enabled_widget.stateChanged.connect(
                partial(self.set_brain_state_enabled, digit)
            )
            state.name_widget.editingFinished.connect(self.check_config_validity)
            state.is_scored_widget.stateChanged.connect(
                partial(self.is_scored_changed, digit)
            )
            state.frequency_widget.valueChanged.connect(self.check_config_validity)

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

    def emg_filter_order_changed(self, new_value: int) -> None:
        """Called when user modifies EMG filter order

        :param new_value: new EMG filter order
        """
        self.emg_filter.order = new_value

    def emg_filter_bp_lower_changed(self, new_value: int | float) -> None:
        """Called when user modifies EMG filter lower cutoff

        :param new_value: new lower bandpass cutoff frequency
        """
        self.emg_filter.bp_lower = new_value
        _ = self.check_config_validity()

    def emg_filter_bp_upper_changed(self, new_value: int | float) -> None:
        """Called when user modifies EMG filter upper cutoff

        :param new_value: new upper bandpass cutoff frequency
        """
        self.emg_filter.bp_upper = new_value
        _ = self.check_config_validity()

    def hyperparameters_changed(self, new_value) -> None:
        """Called when user modifies model training hyperparameters

        :param new_value: unused
        """
        self.hyperparameters = Hyperparameters(
            batch_size=self.ui.batch_size_spinbox.value(),
            learning_rate=self.ui.learning_rate_spinbox.value(),
            momentum=self.ui.momentum_spinbox.value(),
            training_epochs=self.ui.training_epochs_spinbox.value(),
        )

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

        # check validity of EMG filter settings
        if self.emg_filter.bp_lower >= self.emg_filter.bp_upper:
            message = "Error: EMG filter cutoff frequencies are invalid"

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
        save_config(
            brain_state_set=self.brain_state_set,
            default_epoch_length=self.ui.default_epoch_input.value(),
            overwrite_setting=self.ui.overwrite_default_checkbox.isChecked(),
            save_confidence_setting=self.ui.confidence_setting_checkbox.isChecked(),
            min_bout_length=self.ui.default_min_bout_length_spinbox.value(),
            emg_filter=EMGFilter(
                order=self.emg_filter.order,
                bp_lower=self.emg_filter.bp_lower,
                bp_upper=self.emg_filter.bp_upper,
            ),
            hyperparameters=Hyperparameters(
                batch_size=self.hyperparameters.batch_size,
                learning_rate=self.hyperparameters.learning_rate,
                momentum=self.hyperparameters.momentum,
                training_epochs=self.hyperparameters.training_epochs,
            ),
        )
        self.ui.save_config_status.setText("configuration saved")

    def reset_emg_filter_settings(self) -> None:
        self.ui.emg_order_spinbox.setValue(DEFAULT_EMG_FILTER_ORDER)
        self.ui.bp_lower_spinbox.setValue(DEFAULT_EMG_BP_LOWER)
        self.ui.bp_upper_spinbox.setValue(DEFAULT_EMG_BP_UPPER)

    def reset_hyperparams_settings(self):
        self.ui.batch_size_spinbox.setValue(DEFAULT_BATCH_SIZE)
        self.ui.learning_rate_spinbox.setValue(DEFAULT_LEARNING_RATE)
        self.ui.momentum_spinbox.setValue(DEFAULT_MOMENTUM)
        self.ui.training_epochs_spinbox.setValue(DEFAULT_TRAINING_EPOCHS)


def run_primary_window() -> None:
    app = QApplication(sys.argv)
    AccuSleepWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_primary_window()
