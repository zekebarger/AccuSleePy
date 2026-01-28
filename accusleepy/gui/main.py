# AccuSleePy main window
# Icon source: Arkinasi, https://www.flaticon.com/authors/arkinasi

import logging
import os
import sys
from dataclasses import dataclass
from functools import partial

import numpy as np
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
    QLabel,
    QMainWindow,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from accusleepy.brain_state_set import BRAIN_STATES_KEY
from accusleepy.constants import (
    CALIBRATION_FILE_TYPE,
    DEFAULT_MODEL_TYPE,
    LABEL_FILE_TYPE,
    MESSAGE_BOX_MAX_DEPTH,
    MODEL_FILE_TYPE,
    REAL_TIME_MODEL_TYPE,
    RECORDING_FILE_TYPES,
    RECORDING_LIST_FILE_TYPE,
    UNDEFINED_LABEL,
)
from accusleepy.fileio import (
    load_config,
    load_labels,
    load_recording,
    get_version,
)
from accusleepy.gui.dialogs import select_existing_file, select_save_location
from accusleepy.gui.manual_scoring import ManualScoringWindow
from accusleepy.gui.primary_window import Ui_PrimaryWindow
from accusleepy.gui.recording_manager import RecordingListManager
from accusleepy.gui.settings_widget import SettingsWidget
from accusleepy.services import (
    LoadedModel,
    TrainingService,
    check_single_file_inputs,
    create_calibration,
    score_recording_list,
)
from accusleepy.validation import validate_and_correct_labels
from accusleepy.signal_processing import resample_and_standardize
from accusleepy.validation import check_config_consistency

logger = logging.getLogger(__name__)

# on Windows, prevent dark mode from changing the visual style
if os.name == "nt":
    sys.argv += ["-platform", "windows:darkmode=0"]


# relative path to user manual
MAIN_GUIDE_FILE = os.path.normpath(r"text/main_guide.md")


@dataclass
class TrainingSettings:
    """Settings for training a new model"""

    epochs_per_img: int = 9
    delete_images: bool = True
    model_type: str = DEFAULT_MODEL_TYPE
    calibrate: bool = True


@dataclass
class ScoringSettings:
    """Settings for scoring a recording"""

    only_overwrite_undefined: bool
    save_confidence_scores: bool
    min_bout_length: int | float


class AccuSleepWindow(QMainWindow):
    """AccuSleePy primary window"""

    def __init__(self):
        super(AccuSleepWindow, self).__init__()

        # initialize the UI
        self.ui = Ui_PrimaryWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy")

        # Load configuration
        loaded_config = load_config()

        # Apply default values from the configuration
        self.epoch_length = loaded_config.default_epoch_length
        self.scoring = ScoringSettings(
            only_overwrite_undefined=loaded_config.overwrite_setting,
            save_confidence_scores=loaded_config.save_confidence_setting,
            min_bout_length=loaded_config.min_bout_length,
        )

        # Initialize settings tab (manages Settings tab UI and saved config values)
        self.config = SettingsWidget(ui=self.ui, config=loaded_config, parent=self)

        # initialize info about the recordings, classification data / settings
        self.ui.epoch_length_input.setValue(self.epoch_length)
        self.ui.overwritecheckbox.setChecked(self.scoring.only_overwrite_undefined)
        self.ui.save_confidence_checkbox.setChecked(self.scoring.save_confidence_scores)
        self.ui.bout_length_input.setValue(self.scoring.min_bout_length)

        # loaded classification model and its metadata
        self.loaded_model = LoadedModel()

        # settings for training new models
        self.training = TrainingSettings()

        # set up the list of recordings
        self.recording_manager = RecordingListManager(
            self.ui.recording_list_widget, parent=self
        )

        # messages to display
        self.messages = []

        # display current version
        self.ui.version_label.setText(f"v{get_version()}")

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
        self.ui.sampling_rate_input.valueChanged.connect(
            lambda v: setattr(self.recording_manager.current, "sampling_rate", v)
        )
        self.ui.epoch_length_input.valueChanged.connect(
            lambda v: setattr(self, "epoch_length", v)
        )
        self.ui.recording_file_button.clicked.connect(self.select_recording_file)
        self.ui.select_label_button.clicked.connect(self.select_label_file)
        self.ui.create_label_button.clicked.connect(self.create_label_file)
        self.ui.manual_scoring_button.clicked.connect(self.manual_scoring)
        self.ui.create_calibration_button.clicked.connect(self.create_calibration_file)
        self.ui.select_calibration_button.clicked.connect(self.select_calibration_file)
        self.ui.load_model_button.clicked.connect(partial(self.load_model, None))
        self.ui.score_all_button.clicked.connect(self.score_recordings)
        self.ui.overwritecheckbox.stateChanged.connect(
            lambda v: setattr(self.scoring, "only_overwrite_undefined", bool(v))
        )
        self.ui.save_confidence_checkbox.stateChanged.connect(
            lambda v: setattr(self.scoring, "save_confidence_scores", bool(v))
        )
        self.ui.bout_length_input.valueChanged.connect(
            lambda v: setattr(self.scoring, "min_bout_length", v)
        )
        self.ui.user_manual_button.clicked.connect(self.show_user_manual)
        self.ui.image_number_input.valueChanged.connect(
            lambda v: setattr(self.training, "epochs_per_img", v)
        )
        self.ui.delete_image_box.stateChanged.connect(
            lambda v: setattr(self.training, "delete_images", bool(v))
        )
        self.ui.calibrate_checkbox.stateChanged.connect(
            self.update_training_calibration
        )
        self.ui.train_model_button.clicked.connect(self.train_model)
        self.ui.save_config_button.clicked.connect(self.config.save_config)
        self.ui.export_button.clicked.connect(self.export_recording_list)
        self.ui.import_button.clicked.connect(self.import_recording_list)
        self.ui.default_type_button.toggled.connect(self.model_type_radio_buttons)
        self.ui.reset_emg_params_button.clicked.connect(self.config.reset_emg_filter)
        self.ui.reset_hyperparams_button.clicked.connect(self.config.reset_hyperparams)

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
        self.training.model_type = (
            DEFAULT_MODEL_TYPE if default_selected else REAL_TIME_MODEL_TYPE
        )

    def export_recording_list(self) -> None:
        """Save current list of recordings to file"""
        filename = select_save_location(
            self, "Save list of recordings as", "*" + RECORDING_LIST_FILE_TYPE
        )
        if not filename:
            return
        self.recording_manager.export_to_file(filename)
        self.show_message(f"Saved list of recordings to {filename}")

    def import_recording_list(self):
        """Load list of recordings from file, overwriting current list"""
        filename = select_existing_file(
            self, "Select list of recordings", "*" + RECORDING_LIST_FILE_TYPE
        )
        if not filename:
            return

        self.recording_manager.import_from_file(filename)
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
        filename = str(filename)

        _, file_extension = os.path.splitext(filename)

        if obj == self.ui.recording_file_label:
            if file_extension in RECORDING_FILE_TYPES:
                self.recording_manager.current.recording_file = filename
                self.ui.recording_file_label.setText(filename)
        elif obj == self.ui.label_file_label:
            if file_extension == LABEL_FILE_TYPE:
                self.recording_manager.current.label_file = filename
                self.ui.label_file_label.setText(filename)
        elif obj == self.ui.calibration_file_label:
            if file_extension == CALIBRATION_FILE_TYPE:
                self.recording_manager.current.calibration_file = filename
                self.ui.calibration_file_label.setText(filename)
        elif obj == self.ui.model_label:
            self.load_model(filename=filename)

        return super().eventFilter(obj, event)

    def train_model(self) -> None:
        """Train a classification model using the current recordings."""
        model_filename = select_save_location(
            self, "Save classification model file as", "*" + MODEL_FILE_TYPE
        )
        if not model_filename:
            self.show_message("Model training canceled, no filename given")
            return

        # Determine calibration fraction
        if self.training.calibrate:
            calibration_fraction = self.ui.calibration_spinbox.value() / 100
        else:
            calibration_fraction = 0

        # Show progress message
        self.show_message("Training, please wait. See console for progress updates.")
        self.ui.message_area.repaint()
        QApplication.processEvents()

        # Create service and run training
        service = TrainingService(progress_callback=self.show_message)
        result = service.train_model(
            recordings=list(self.recording_manager),
            epoch_length=self.epoch_length,
            epochs_per_img=self.training.epochs_per_img,
            model_type=self.training.model_type,
            calibrate=self.training.calibrate,
            calibration_fraction=calibration_fraction,
            brain_state_set=self.config.brain_state_set,
            emg_filter=self.config.emg_filter,
            hyperparameters=self.config.hyperparameters,
            model_filename=model_filename,
            delete_images=self.training.delete_images,
        )

        # Display results
        result.report_to(self.show_message)

    def update_training_calibration(self) -> None:
        """Update choice of whether to calibrate model after training"""
        self.training.calibrate = self.ui.calibrate_checkbox.isChecked()
        self.ui.calibration_spinbox.setEnabled(self.training.calibrate)

    def score_recordings(self) -> None:
        """Score all recordings using the classification model."""
        self.ui.score_all_status.setText("running...")
        self.ui.score_all_status.repaint()
        QApplication.processEvents()

        result = score_recording_list(
            recordings=list(self.recording_manager),
            loaded_model=self.loaded_model,
            epoch_length=self.epoch_length,
            only_overwrite_undefined=self.scoring.only_overwrite_undefined,
            save_confidence_scores=self.scoring.save_confidence_scores,
            min_bout_length=self.scoring.min_bout_length,
            brain_state_set=self.config.brain_state_set,
            emg_filter=self.config.emg_filter,
        )

        # Display results
        result.report_to(self.show_message)
        self.ui.score_all_status.setText("error" if not result.success else "")

    def load_model(self, filename=None) -> None:
        """Load trained classification model from file

        :param filename: model filename, if it's known
        """
        if filename is None:
            filename = select_existing_file(
                self, "Select classification model", "*" + MODEL_FILE_TYPE
            )
            if not filename:
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
            logger.exception("Failed to load %s", filename)
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

        self.loaded_model.model = model
        self.loaded_model.epoch_length = epoch_length
        self.loaded_model.epochs_per_img = epochs_per_img

        # warn user if the model's expected epoch length or brain states
        # don't match the current configuration
        config_warnings = check_config_consistency(
            current_brain_states=self.config.brain_state_set.to_output_dict()[
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
        error_message = check_single_file_inputs(
            self.recording_manager.current, self.epoch_length
        )
        if error_message:
            status_widget.setText(error_message)
            self.show_message(f"ERROR: {error_message}")
            return None, None, None, False

        try:
            eeg, emg = load_recording(self.recording_manager.current.recording_file)
        except Exception:
            logger.exception(
                "Failed to load %s",
                self.recording_manager.current.recording_file,
            )
            status_widget.setText("could not load recording")
            self.show_message(
                (
                    "ERROR: could not load recording. "
                    "Check user manual for formatting instructions."
                )
            )
            return None, None, None, False

        sampling_rate = self.recording_manager.current.sampling_rate

        eeg, emg, sampling_rate = resample_and_standardize(
            eeg=eeg,
            emg=emg,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
        )

        return eeg, emg, sampling_rate, True

    def create_calibration_file(self) -> None:
        """Creates a calibration file.

        This loads a recording and its labels, checks that the labels are
        all valid, creates the calibration file, and sets the
        "calibration file" property of the current recording to be the
        newly created file.
        """
        filename = select_save_location(
            self, "Save calibration file as", "*" + CALIBRATION_FILE_TYPE
        )
        if not filename:
            return

        result = create_calibration(
            recording=self.recording_manager.current,
            epoch_length=self.epoch_length,
            brain_state_set=self.config.brain_state_set,
            emg_filter=self.config.emg_filter,
            output_filename=filename,
        )

        # Display results
        result.report_to(self.show_message)
        if not result.success:
            self.ui.calibration_status.setText("error")
        else:
            self.ui.calibration_status.setText("")
            self.recording_manager.current.calibration_file = filename
            self.ui.calibration_file_label.setText(filename)

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
        label_file = self.recording_manager.current.label_file
        if os.path.isfile(label_file):
            try:
                labels, confidence_scores = load_labels(label_file)
            except Exception:
                logger.exception("Failed to load %s", label_file)
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

        # check that labels are valid and correct minor length mismatches
        labels, confidence_scores, validation_message = validate_and_correct_labels(
            labels=labels,
            confidence_scores=confidence_scores,
            samples_in_recording=eeg.size,
            sampling_rate=sampling_rate,
            epoch_length=self.epoch_length,
            brain_state_set=self.config.brain_state_set,
        )
        if labels is None:
            self.ui.manual_scoring_status.setText("invalid label file")
            self.show_message(f"ERROR: {validation_message}")
            return
        if validation_message:
            self.show_message(f"WARNING: {validation_message}")

        self.show_message(f"Viewing recording {self.recording_manager.current.name}")
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
            emg_filter=self.config.emg_filter,
        )
        manual_scoring_window.setWindowTitle(f"AccuSleePy viewer: {label_file}")
        manual_scoring_window.exec()
        self.ui.manual_scoring_status.setText("")

    def create_label_file(self) -> None:
        """Set the filename for a new label file"""
        filename = select_save_location(
            self,
            "Set filename for label file (nothing will be overwritten yet)",
            "*" + LABEL_FILE_TYPE,
        )
        if filename:
            self.recording_manager.current.label_file = filename
            self.ui.label_file_label.setText(filename)

    def select_label_file(self) -> None:
        """User can select an existing label file"""
        filename = select_existing_file(
            self, "Select label file", "*" + LABEL_FILE_TYPE
        )
        if filename:
            self.recording_manager.current.label_file = filename
            self.ui.label_file_label.setText(filename)

    def select_calibration_file(self) -> None:
        """User can select a calibration file"""
        filename = select_existing_file(
            self, "Select calibration file", "*" + CALIBRATION_FILE_TYPE
        )
        if filename:
            self.recording_manager.current.calibration_file = filename
            self.ui.calibration_file_label.setText(filename)

    def select_recording_file(self) -> None:
        """User can select a recording file"""
        file_filter = f"(*{' *'.join(RECORDING_FILE_TYPES)})"
        filename = select_existing_file(self, "Select recording file", file_filter)
        if filename:
            self.recording_manager.current.recording_file = filename
            self.ui.recording_file_label.setText(filename)

    def show_recording_info(self) -> None:
        """Update the UI to show info for the selected recording"""
        recording = self.recording_manager.current
        self.ui.sampling_rate_input.setValue(recording.sampling_rate)
        self.ui.recording_file_label.setText(recording.recording_file)
        self.ui.label_file_label.setText(recording.label_file)
        self.ui.calibration_file_label.setText(recording.calibration_file)

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

    def select_recording(self, _index: int) -> None:
        """Callback for when a recording is selected"""
        self.show_recording_info()
        self.ui.selected_recording_groupbox.setTitle(
            f"Data / actions for Recording {self.recording_manager.current.name}"
        )

    def add_recording(self) -> None:
        """Add new recording to the list"""
        current_sampling_rate = self.recording_manager.current.sampling_rate
        recording = self.recording_manager.add(sampling_rate=current_sampling_rate)
        self.show_message(f"added Recording {recording.name}")

    def remove_recording(self) -> None:
        """Delete selected recording from the list"""
        self.show_message(self.recording_manager.remove_current())

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


def run_primary_window() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )
    app = QApplication(sys.argv)
    AccuSleepWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_primary_window()
