"""GUI tests for AccuSleepWindow interactions.

Tests for:
1. Scoring settings
2. Training settings
3. Message display
4. Recording info display
5. Recording list import/export
6. File path assignment
"""

import pytest

from accusleepy.constants import DEFAULT_MODEL_TYPE, REAL_TIME_MODEL_TYPE
from accusleepy.gui.main import AccuSleepWindow


@pytest.mark.gui
class TestScoringSettings:
    """Tests for scoring-related settings in the main window."""

    def test_overwrite_checkbox_updates_setting(self, qtbot):
        """Toggling the overwrite checkbox works."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        initial = window.scoring.only_overwrite_undefined
        window.ui.overwritecheckbox.setChecked(not initial)
        assert window.scoring.only_overwrite_undefined != initial
        window.close()

    def test_save_confidence_checkbox_updates_setting(self, qtbot):
        """Toggling the confidence checkbox works."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        initial = window.scoring.save_confidence_scores
        window.ui.save_confidence_checkbox.setChecked(not initial)
        assert window.scoring.save_confidence_scores != initial
        window.close()

    def test_bout_length_input_updates_setting(self, qtbot):
        """Changing the minimum bout length works."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.ui.bout_length_input.setValue(10)
        assert window.scoring.min_bout_length == 10

        window.ui.bout_length_input.setValue(12)
        assert window.scoring.min_bout_length == 12
        window.close()


@pytest.mark.gui
class TestTrainingSettings:
    """Tests for training-related settings in the main window."""

    def test_model_type_radio_buttons(self, qtbot):
        """Model type radio buttons work."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        # Default type should be selected initially
        assert window.training.model_type == DEFAULT_MODEL_TYPE

        # Select real-time type
        window.ui.real_time_button.setChecked(True)
        assert window.training.model_type == REAL_TIME_MODEL_TYPE

        # Back to default
        window.ui.default_type_button.setChecked(True)
        assert window.training.model_type == DEFAULT_MODEL_TYPE
        window.close()

    def test_epochs_per_image_input(self, qtbot):
        """Changing epochs per image setting works."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.ui.image_number_input.setValue(15)
        assert window.training.epochs_per_img == 15
        window.ui.image_number_input.setValue(17)
        assert window.training.epochs_per_img == 17
        window.close()

    def test_calibration_checkbox_toggles_spinbox(self, qtbot):
        """Toggling calibration checkbox enables/disables the spinbox."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        # Check initial state
        window.ui.calibrate_checkbox.setChecked(True)
        assert window.training.calibrate is True
        assert window.ui.calibration_spinbox.isEnabled()

        # Uncheck disables spinbox
        window.ui.calibrate_checkbox.setChecked(False)
        assert window.training.calibrate is False
        assert not window.ui.calibration_spinbox.isEnabled()

        # Re-check enables spinbox
        window.ui.calibrate_checkbox.setChecked(True)
        assert window.training.calibrate is True
        assert window.ui.calibration_spinbox.isEnabled()
        window.close()


@pytest.mark.gui
class TestMessageDisplay:
    """Tests for the message area in the main window."""

    def test_show_message_appends_text(self, qtbot):
        """show_message appends to the message area."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.show_message("hello")
        assert "hello" in window.ui.message_area.toPlainText()

        window.show_message("world")
        text = window.ui.message_area.toPlainText()
        assert "hello" in text
        assert "world" in text
        window.close()

    def test_show_message_tracks_history(self, qtbot):
        """Messages are stored in the messages list."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.show_message("first")
        window.show_message("second")
        assert window.messages == ["first", "second"]
        window.close()

    def test_add_recording_shows_message(self, qtbot):
        """Adding a recording displays a message."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.add_recording()
        assert any("added Recording" in m for m in window.messages)
        window.close()

    def test_remove_recording_shows_message(self, qtbot):
        """Removing a recording displays a message."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.remove_recording()
        assert any(
            "cleared Recording" in m or "deleted Recording" in m
            for m in window.messages
        )
        window.close()


@pytest.mark.gui
class TestRecordingInfoDisplay:
    """Tests for recording info display in the UI."""

    def test_file_paths_shown_for_selected_recording(self, qtbot):
        """Selecting a recording updates file path labels in the UI."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        # Set files on first recording
        window.recording_manager.current.recording_file = "/path/to/rec.csv"
        window.recording_manager.current.label_file = "/path/to/labels.csv"
        window.recording_manager.current.calibration_file = "/path/to/calib.mat"

        # Add second recording with different files
        window.add_recording()
        window.recording_manager.current.recording_file = "/other/rec.csv"
        window.recording_manager.current.label_file = "/other/labels.csv"

        # Switch back to first recording
        window.ui.recording_list_widget.setCurrentRow(0)

        assert window.ui.recording_file_label.text() == "/path/to/rec.csv"
        assert window.ui.label_file_label.text() == "/path/to/labels.csv"
        assert window.ui.calibration_file_label.text() == "/path/to/calib.mat"

        # Switch to second recording
        window.ui.recording_list_widget.setCurrentRow(1)

        assert window.ui.recording_file_label.text() == "/other/rec.csv"
        assert window.ui.label_file_label.text() == "/other/labels.csv"
        window.close()

    def test_epoch_length_input_updates_property(self, qtbot):
        """Changing the epoch length input updates the epoch_length property."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.ui.epoch_length_input.setValue(10)
        assert window.epoch_length == 10
        window.ui.epoch_length_input.setValue(12)
        assert window.epoch_length == 12
        window.close()


@pytest.mark.gui
class TestRecordingListImportExport:
    """Tests for recording list import/export functionality."""

    def test_export_and_import_round_trip(self, qtbot, tmp_path):
        """Exporting and re-importing a recording list preserves data."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        # Set up recordings
        window.ui.sampling_rate_input.setValue(128)
        window.recording_manager.current.recording_file = "/data/rec1.csv"
        window.recording_manager.current.label_file = "/data/labels1.csv"

        window.add_recording()
        window.ui.sampling_rate_input.setValue(256)
        window.recording_manager.current.recording_file = "/data/rec2.csv"

        assert len(window.recording_manager) == 2

        # Export
        export_file = str(tmp_path / "recordings.json")
        window.recording_manager.export_to_file(export_file)

        # Create a fresh window and import
        window2 = AccuSleepWindow()
        qtbot.addWidget(window2)
        assert len(window2.recording_manager) == 1

        window2.recording_manager.import_from_file(export_file)
        assert len(window2.recording_manager) == 2

        # Check first recording's data
        window2.ui.recording_list_widget.setCurrentRow(0)
        assert window2.recording_manager.current.sampling_rate == 128
        assert window2.recording_manager.current.recording_file == "/data/rec1.csv"
        assert window2.recording_manager.current.label_file == "/data/labels1.csv"

        # Check second recording's data
        window2.ui.recording_list_widget.setCurrentRow(1)
        assert window2.recording_manager.current.sampling_rate == 256
        assert window2.recording_manager.current.recording_file == "/data/rec2.csv"

        window.close()
        window2.close()
