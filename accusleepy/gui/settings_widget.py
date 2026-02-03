"""Settings tab manager"""

from dataclasses import dataclass
from functools import partial

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QLineEdit

from accusleepy.brain_state_set import BrainState, BrainStateSet
from accusleepy.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DELETE_TRAINING_IMAGES_STATE,
    DEFAULT_EMG_BP_LOWER,
    DEFAULT_EMG_BP_UPPER,
    DEFAULT_EMG_FILTER_ORDER,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOMENTUM,
    DEFAULT_TRAINING_EPOCHS,
    UNDEFINED_LABEL,
)
from accusleepy.fileio import (
    AccuSleePyConfig,
    EMGFilter,
    Hyperparameters,
    save_config,
)
from accusleepy.gui.primary_window import Ui_PrimaryWindow


@dataclass
class StateSettings:
    """Widgets for config settings for a brain state"""

    digit: int
    enabled_widget: QCheckBox
    name_widget: QLineEdit
    is_scored_widget: QCheckBox
    frequency_widget: QDoubleSpinBox


class SettingsWidget(QObject):
    """Manages settings tab UI and configuration data"""

    def __init__(
        self,
        ui: Ui_PrimaryWindow,
        config: AccuSleePyConfig,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._ui = ui

        # Store configuration values (only settings managed by the Settings tab)
        self._brain_state_set = config.brain_state_set
        self._emg_filter = config.emg_filter
        self._hyperparameters = config.hyperparameters
        self._default_epochs_to_show = config.epochs_to_show
        self._default_autoscroll_state = config.autoscroll_state
        self._delete_training_images = config.delete_training_images

        # Store default values for main tab settings (used to populate Settings tab UI)
        self._default_epoch_length = config.default_epoch_length
        self._default_overwrite_setting = config.overwrite_setting
        self._default_confidence_setting = config.save_confidence_setting
        self._default_min_bout_length = config.min_bout_length

        # Initialize settings widgets
        self._settings_widgets: dict[int, StateSettings] = {}
        self._initialize_settings_tab()

    @property
    def brain_state_set(self) -> BrainStateSet:
        """The current brain state set configuration (from saved config)"""
        return self._brain_state_set

    @property
    def emg_filter(self) -> EMGFilter:
        """EMG filter parameters (from saved config)"""
        return self._emg_filter

    @property
    def hyperparameters(self) -> Hyperparameters:
        """Model training hyperparameters"""
        return self._hyperparameters

    @property
    def delete_training_images(self) -> bool:
        """Whether to delete images after training"""
        return self._delete_training_images

    @property
    def default_epochs_to_show(self) -> int:
        """Default number of epochs to show in manual scoring"""
        return self._default_epochs_to_show

    @property
    def default_autoscroll_state(self) -> bool:
        """Default autoscroll state for manual scoring"""
        return self._default_autoscroll_state

    def _initialize_settings_tab(self) -> None:
        """Populate settings tab and assign its callbacks"""
        # Store dictionary that maps digits to rows of widgets
        self._settings_widgets = {
            1: StateSettings(
                digit=1,
                enabled_widget=self._ui.enable_state_1,
                name_widget=self._ui.state_name_1,
                is_scored_widget=self._ui.state_scored_1,
                frequency_widget=self._ui.state_frequency_1,
            ),
            2: StateSettings(
                digit=2,
                enabled_widget=self._ui.enable_state_2,
                name_widget=self._ui.state_name_2,
                is_scored_widget=self._ui.state_scored_2,
                frequency_widget=self._ui.state_frequency_2,
            ),
            3: StateSettings(
                digit=3,
                enabled_widget=self._ui.enable_state_3,
                name_widget=self._ui.state_name_3,
                is_scored_widget=self._ui.state_scored_3,
                frequency_widget=self._ui.state_frequency_3,
            ),
            4: StateSettings(
                digit=4,
                enabled_widget=self._ui.enable_state_4,
                name_widget=self._ui.state_name_4,
                is_scored_widget=self._ui.state_scored_4,
                frequency_widget=self._ui.state_frequency_4,
            ),
            5: StateSettings(
                digit=5,
                enabled_widget=self._ui.enable_state_5,
                name_widget=self._ui.state_name_5,
                is_scored_widget=self._ui.state_scored_5,
                frequency_widget=self._ui.state_frequency_5,
            ),
            6: StateSettings(
                digit=6,
                enabled_widget=self._ui.enable_state_6,
                name_widget=self._ui.state_name_6,
                is_scored_widget=self._ui.state_scored_6,
                frequency_widget=self._ui.state_frequency_6,
            ),
            7: StateSettings(
                digit=7,
                enabled_widget=self._ui.enable_state_7,
                name_widget=self._ui.state_name_7,
                is_scored_widget=self._ui.state_scored_7,
                frequency_widget=self._ui.state_frequency_7,
            ),
            8: StateSettings(
                digit=8,
                enabled_widget=self._ui.enable_state_8,
                name_widget=self._ui.state_name_8,
                is_scored_widget=self._ui.state_scored_8,
                frequency_widget=self._ui.state_frequency_8,
            ),
            9: StateSettings(
                digit=9,
                enabled_widget=self._ui.enable_state_9,
                name_widget=self._ui.state_name_9,
                is_scored_widget=self._ui.state_scored_9,
                frequency_widget=self._ui.state_frequency_9,
            ),
            0: StateSettings(
                digit=0,
                enabled_widget=self._ui.enable_state_0,
                name_widget=self._ui.state_name_0,
                is_scored_widget=self._ui.state_scored_0,
                frequency_widget=self._ui.state_frequency_0,
            ),
        }

        # Update widget state to display current config
        # UI defaults for main tab settings (shown in Settings tab for configuration)
        self._ui.default_epoch_input.setValue(self._default_epoch_length)
        self._ui.overwrite_default_checkbox.setChecked(self._default_overwrite_setting)
        self._ui.confidence_setting_checkbox.setChecked(
            self._default_confidence_setting
        )
        self._ui.default_min_bout_length_spinbox.setValue(self._default_min_bout_length)
        self._ui.epochs_to_show_spinbox.setValue(self._default_epochs_to_show)
        self._ui.autoscroll_checkbox.setChecked(self._default_autoscroll_state)
        # EMG filter
        self._ui.emg_order_spinbox.setValue(self._emg_filter.order)
        self._ui.bp_lower_spinbox.setValue(self._emg_filter.bp_lower)
        self._ui.bp_upper_spinbox.setValue(self._emg_filter.bp_upper)
        # Model training hyperparameters
        self._ui.batch_size_spinbox.setValue(self._hyperparameters.batch_size)
        self._ui.learning_rate_spinbox.setValue(self._hyperparameters.learning_rate)
        self._ui.momentum_spinbox.setValue(self._hyperparameters.momentum)
        self._ui.training_epochs_spinbox.setValue(self._hyperparameters.training_epochs)
        self._ui.delete_image_box.setChecked(self._delete_training_images)
        # Brain states
        states = {b.digit: b for b in self._brain_state_set.brain_states}
        for digit in range(10):
            if digit in states:
                self._settings_widgets[digit].enabled_widget.setChecked(True)
                self._settings_widgets[digit].name_widget.setText(states[digit].name)
                self._settings_widgets[digit].is_scored_widget.setChecked(
                    states[digit].is_scored
                )
                self._settings_widgets[digit].frequency_widget.setValue(
                    states[digit].frequency
                )
            else:
                self._settings_widgets[digit].enabled_widget.setChecked(False)
                self._settings_widgets[digit].name_widget.setEnabled(False)
                self._settings_widgets[digit].is_scored_widget.setEnabled(False)
                self._settings_widgets[digit].frequency_widget.setEnabled(False)

        # Set callbacks
        self._ui.default_epoch_input.valueChanged.connect(self.reset_status_message)
        self._ui.overwrite_default_checkbox.stateChanged.connect(
            self.reset_status_message
        )
        self._ui.confidence_setting_checkbox.stateChanged.connect(
            self.reset_status_message
        )
        self._ui.default_min_bout_length_spinbox.valueChanged.connect(
            self.reset_status_message
        )
        self._ui.epochs_to_show_spinbox.valueChanged.connect(self.reset_status_message)
        self._ui.autoscroll_checkbox.stateChanged.connect(self.reset_status_message)

        self._ui.emg_order_spinbox.valueChanged.connect(self.reset_status_message)
        self._ui.bp_lower_spinbox.valueChanged.connect(
            self._emg_filter_bp_lower_changed
        )
        self._ui.bp_upper_spinbox.valueChanged.connect(
            self._emg_filter_bp_upper_changed
        )
        self._ui.batch_size_spinbox.valueChanged.connect(self._hyperparameters_changed)
        self._ui.learning_rate_spinbox.valueChanged.connect(
            self._hyperparameters_changed
        )
        self._ui.momentum_spinbox.valueChanged.connect(self._hyperparameters_changed)
        self._ui.training_epochs_spinbox.valueChanged.connect(
            self._hyperparameters_changed
        )
        self._ui.delete_image_box.stateChanged.connect(self._hyperparameters_changed)
        for digit in range(10):
            state = self._settings_widgets[digit]
            state.enabled_widget.stateChanged.connect(
                partial(self._set_brain_state_enabled, digit)
            )
            state.name_widget.editingFinished.connect(self.check_validity)
            state.is_scored_widget.stateChanged.connect(
                partial(self._is_scored_changed, digit)
            )
            state.frequency_widget.valueChanged.connect(self.check_validity)

    def _set_brain_state_enabled(self, digit: int, _state: int) -> None:
        """Called when user clicks "enabled" checkbox

        :param digit: brain state digit
        :param _state: unused but mandatory
        """
        state = self._settings_widgets[digit]
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
        self.check_validity()

    def _is_scored_changed(self, digit: int, _state: int) -> None:
        """Called when user sets whether a state is scored

        :param digit: brain state digit
        :param _state: unused but mandatory
        """
        state = self._settings_widgets[digit]
        is_checked = state.is_scored_widget.isChecked()
        state.frequency_widget.setEnabled(is_checked)
        if not is_checked:
            state.frequency_widget.setValue(0)
        self.check_validity()

    def _emg_filter_bp_lower_changed(self, _new_value: int | float) -> None:
        """Called when user modifies EMG filter lower cutoff

        Value is read from UI when config is saved. Triggers validation.
        """
        self.check_validity()

    def _emg_filter_bp_upper_changed(self, _new_value: int | float) -> None:
        """Called when user modifies EMG filter upper cutoff

        Value is read from UI when config is saved. Triggers validation.
        """
        self.check_validity()

    def _hyperparameters_changed(self, _new_value) -> None:
        """Called when user modifies model training hyperparameters"""
        # These are the only changes to the settings tab that take
        # effect immediately
        self._hyperparameters = Hyperparameters(
            batch_size=self._ui.batch_size_spinbox.value(),
            learning_rate=self._ui.learning_rate_spinbox.value(),
            momentum=self._ui.momentum_spinbox.value(),
            training_epochs=self._ui.training_epochs_spinbox.value(),
        )
        self._delete_training_images = self._ui.delete_image_box.isChecked()
        self._ui.save_config_status.setText("")

    def reset_status_message(self, _new_value=None) -> None:
        """Clear the message next to the 'save' button"""
        self._ui.save_config_status.setText("")

    def check_validity(self) -> str | None:
        """Check if brain state configuration on screen is valid

        :return: error message if invalid, None if valid
        """
        message = None

        # Strip whitespace from brain state names and update display
        for digit in range(10):
            state = self._settings_widgets[digit]
            current_name = state.name_widget.text()
            formatted_name = current_name.strip()
            if current_name != formatted_name:
                state.name_widget.setText(formatted_name)

        # Check if names are unique and frequencies add up to 1
        names = []
        frequencies = []
        for digit in range(10):
            state = self._settings_widgets[digit]
            if state.enabled_widget.isChecked():
                names.append(state.name_widget.text())
                frequencies.append(state.frequency_widget.value())
        if len(names) != len(set(names)):
            message = "Error: names must be unique"
        if sum(frequencies) != 1:
            message = "Error: sum(frequencies) != 1"

        # Check validity of EMG filter settings (read from UI, not saved state)
        bp_lower = self._ui.bp_lower_spinbox.value()
        bp_upper = self._ui.bp_upper_spinbox.value()
        if bp_lower >= bp_upper:
            message = "Error: EMG filter cutoff frequencies are invalid"

        if message is not None:
            self._ui.save_config_status.setText(message)
            self._ui.save_config_button.setEnabled(False)
            return message

        self._ui.save_config_button.setEnabled(True)
        self._ui.save_config_status.setText("")
        return None

    def save_config(self) -> None:
        """Save configuration to file"""
        # Check that configuration is valid
        error_message = self.check_validity()
        if error_message is not None:
            return

        # Build a BrainStateSet object from the current configuration
        brain_states = []
        for digit in range(10):
            state = self._settings_widgets[digit]
            if state.enabled_widget.isChecked():
                brain_states.append(
                    BrainState(
                        name=state.name_widget.text(),
                        digit=digit,
                        is_scored=state.is_scored_widget.isChecked(),
                        frequency=state.frequency_widget.value(),
                    )
                )

        # Update brain state set and EMG filter from UI
        # Note that this only occurs when a valid configuration is saved
        self._brain_state_set = BrainStateSet(brain_states, UNDEFINED_LABEL)
        self._emg_filter = EMGFilter(
            order=self._ui.emg_order_spinbox.value(),
            bp_lower=self._ui.bp_lower_spinbox.value(),
            bp_upper=self._ui.bp_upper_spinbox.value(),
        )

        # Save to file
        save_config(
            brain_state_set=self._brain_state_set,
            default_epoch_length=self._ui.default_epoch_input.value(),
            overwrite_setting=self._ui.overwrite_default_checkbox.isChecked(),
            save_confidence_setting=self._ui.confidence_setting_checkbox.isChecked(),
            min_bout_length=self._ui.default_min_bout_length_spinbox.value(),
            emg_filter=self._emg_filter,
            hyperparameters=self._hyperparameters,
            epochs_to_show=self._ui.epochs_to_show_spinbox.value(),
            autoscroll_state=self._ui.autoscroll_checkbox.isChecked(),
            delete_training_images=self._ui.delete_image_box.isChecked(),
        )
        self._ui.save_config_status.setText("configuration saved")

    def reset_emg_filter(self) -> None:
        """Reset EMG filter settings to defaults"""
        self._ui.emg_order_spinbox.setValue(DEFAULT_EMG_FILTER_ORDER)
        self._ui.bp_lower_spinbox.setValue(DEFAULT_EMG_BP_LOWER)
        self._ui.bp_upper_spinbox.setValue(DEFAULT_EMG_BP_UPPER)

    def reset_hyperparams(self) -> None:
        """Reset hyperparameters to defaults"""
        self._ui.batch_size_spinbox.setValue(DEFAULT_BATCH_SIZE)
        self._ui.learning_rate_spinbox.setValue(DEFAULT_LEARNING_RATE)
        self._ui.momentum_spinbox.setValue(DEFAULT_MOMENTUM)
        self._ui.training_epochs_spinbox.setValue(DEFAULT_TRAINING_EPOCHS)
        self._ui.delete_image_box.setChecked(DEFAULT_DELETE_TRAINING_IMAGES_STATE)
