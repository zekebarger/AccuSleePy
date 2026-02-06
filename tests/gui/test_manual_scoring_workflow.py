"""GUI tests for manual scoring workflow.

Tests for:
1. Main window -> manual scoring integration
2. Label modification in manual scoring
3. Undo/redo functionality
4. Recording list management
5. Bout navigation (jump to next state)
6. Save functionality
"""

import numpy as np
import pytest

from accusleepy.constants import UNDEFINED_LABEL
from accusleepy.gui.main import AccuSleepWindow
from accusleepy.gui.manual_scoring import ManualScoringWindow


def sync_and_close(window):
    """Sync labels and close window to avoid the 'unsaved changes' dialog."""
    window.last_saved_labels = window.labels.copy()
    window.close()


@pytest.mark.gui
class TestMainWindowToManualScoring:
    """Main window can launch the manual scoring window."""

    def test_launch_manual_scoring(
        self,
        qtbot,
        monkeypatch,
        synthetic_recording_file_for_gui,
        synthetic_label_file_for_gui,
        sample_eeg_emg_for_viewer,
    ):
        """Main window can set sampling rate, files, and launch manual scoring."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        # Set sampling rate and epoch length to match synthetic data
        target_sampling_rate = sample_eeg_emg_for_viewer["sampling_rate"]
        target_epoch_length = sample_eeg_emg_for_viewer["epoch_length"]
        window.ui.sampling_rate_input.setValue(target_sampling_rate)
        window.ui.epoch_length_input.setValue(target_epoch_length)
        assert window.recording_manager.current.sampling_rate == target_sampling_rate
        assert window.epoch_length == target_epoch_length

        # Set recording and label files directly
        window.recording_manager.current.recording_file = str(
            synthetic_recording_file_for_gui
        )
        window.recording_manager.current.label_file = str(synthetic_label_file_for_gui)

        # Capture the ManualScoringWindow instance
        captured_window = {}

        def mock_exec(msw_self):
            """Capture the window instance without blocking."""
            captured_window["instance"] = msw_self
            return 0

        monkeypatch.setattr(ManualScoringWindow, "exec", mock_exec)

        # Launch manual scoring
        window.manual_scoring()

        # Verify ManualScoringWindow was created with correct parameters
        assert "instance" in captured_window
        msw = captured_window["instance"]
        assert msw.sampling_rate == target_sampling_rate
        assert msw.n_epochs == sample_eeg_emg_for_viewer["n_epochs"]

        window.close()


@pytest.mark.gui
class TestManualScoringLabelModification:
    """Tests for label modification in ManualScoringWindow."""

    def test_modify_current_epoch_label(self, manual_scoring_window):
        """Changing label updates the labels array."""
        window = manual_scoring_window
        assert window.labels[0] == 1

        window.modify_current_epoch_label(2)

        assert window.labels[0] == 2
        sync_and_close(window)

    def test_label_modification_adds_to_history(self, manual_scoring_window):
        """Label changes are tracked in history."""
        window = manual_scoring_window

        assert len(window.history) == 0
        assert window.history_index == 0

        window.modify_current_epoch_label(2)

        assert len(window.history) == 1
        assert window.history_index == 1

        state_change = window.history[0]
        assert state_change.epoch == 0
        assert state_change.previous_labels[0] == 1
        assert state_change.new_labels[0] == 2
        sync_and_close(window)


@pytest.mark.gui
class TestManualScoringUndoRedo:
    """Tests for undo/redo functionality in ManualScoringWindow."""

    def test_undo_single_change(self, manual_scoring_window):
        """Undo reverts a single label change."""
        window = manual_scoring_window

        window.modify_current_epoch_label(2)
        assert window.labels[0] == 2

        window.undo()
        assert window.labels[0] == 1
        sync_and_close(window)

    def test_redo_undone_change(self, manual_scoring_window):
        """Redo restores an undone change."""
        window = manual_scoring_window

        window.modify_current_epoch_label(2)
        window.undo()
        assert window.labels[0] == 1

        window.redo()
        assert window.labels[0] == 2
        sync_and_close(window)

    def test_undo_at_beginning_is_safe(self, manual_scoring_window):
        """Undo with no history doesn't crash."""
        window = manual_scoring_window

        assert len(window.history) == 0
        window.undo()
        assert window.labels[0] == 1
        window.close()

    def test_redo_at_end_is_safe(self, manual_scoring_window):
        """Redo with nothing to redo doesn't crash."""
        window = manual_scoring_window

        window.modify_current_epoch_label(2)
        window.redo()
        assert window.labels[0] == 2
        sync_and_close(window)

    def test_new_change_clears_redo_history(self, manual_scoring_window):
        """A new change after undo clears the redo history."""
        window = manual_scoring_window

        window.modify_current_epoch_label(2)
        window.undo()
        assert window.labels[0] == 1

        window.modify_current_epoch_label(3)

        # Redo should do nothing (redo history was cleared)
        window.redo()
        assert window.labels[0] == 3
        sync_and_close(window)

    def test_multiple_undo_redo(self, manual_scoring_window):
        """Multiple sequential undo/redo operations work correctly."""
        window = manual_scoring_window
        # Initial label at index 0 is 1
        window.modify_current_epoch_label(2)
        window.modify_current_epoch_label(3)
        window.modify_current_epoch_label(UNDEFINED_LABEL)
        assert window.labels[0] == UNDEFINED_LABEL
        assert len(window.history) == 3

        # Undo all 3 changes
        window.undo()
        assert window.labels[0] == 3
        window.undo()
        assert window.labels[0] == 2
        window.undo()
        assert window.labels[0] == 1

        # Redo all 3 changes
        window.redo()
        assert window.labels[0] == 2
        window.redo()
        assert window.labels[0] == 3
        window.redo()
        assert window.labels[0] == UNDEFINED_LABEL
        sync_and_close(window)


@pytest.mark.gui
class TestRecordingListManagement:
    """Tests for recording list management in AccuSleepWindow."""

    def test_add_recording(self, qtbot):
        """Adding a recording increases the count."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        assert len(window.recording_manager) == 1
        window.add_recording()
        assert len(window.recording_manager) == 2
        window.close()

    def test_remove_recording(self, qtbot):
        """Removing a recording decreases the count."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.add_recording()
        assert len(window.recording_manager) == 2

        window.remove_recording()
        assert len(window.recording_manager) == 1
        window.close()

    def test_select_recording_updates_ui(self, qtbot):
        """Selecting a different recording updates the UI."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        window.ui.sampling_rate_input.setValue(128)
        window.add_recording()
        window.ui.sampling_rate_input.setValue(256)
        assert window.recording_manager.current.sampling_rate == 256

        window.ui.recording_list_widget.setCurrentRow(0)
        assert window.ui.sampling_rate_input.value() == 128
        window.close()

    def test_remove_last_recording_resets(self, qtbot):
        """Removing the only recording resets it instead of deleting."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)
        default_sampling_rate = window.recording_manager.current.sampling_rate

        assert len(window.recording_manager) == 1
        window.ui.sampling_rate_input.setValue(default_sampling_rate + 1)
        assert (
            window.recording_manager.current.sampling_rate == default_sampling_rate + 1
        )

        window.remove_recording()

        assert len(window.recording_manager) == 1
        assert window.recording_manager.current.sampling_rate == default_sampling_rate
        window.close()


@pytest.mark.gui
class TestManualScoringNavigation:
    """Tests for epoch navigation in ManualScoringWindow."""

    def test_shift_epoch_navigation(self, manual_scoring_window):
        """Shift epoch navigates forward/backward with bounds checking."""
        window = manual_scoring_window

        assert window.epoch == 0

        window.shift_epoch("right")
        assert window.epoch == 1

        window.shift_epoch("left")
        assert window.epoch == 0

        # Left at epoch 0 should stay at 0
        window.shift_epoch("left")
        assert window.epoch == 0

        # Navigate to last epoch
        for _ in range(window.n_epochs - 1):
            window.shift_epoch("right")
        assert window.epoch == window.n_epochs - 1

        # Right at last epoch should stay there
        window.shift_epoch("right")
        assert window.epoch == window.n_epochs - 1
        window.close()

    def test_jump_to_next_different_state(self, manual_scoring_window):
        """Jump to next epoch with a different brain state."""
        # Labels start with [1, 1, 2, 2, 3, 3, ...]
        window = manual_scoring_window

        assert window.epoch == 0
        window.jump_to_next_state("right", "different")

        # Should land on epoch 2 (first epoch with label != 1)
        assert window.epoch == 2
        sync_and_close(window)

    def test_jump_to_next_undefined_state(
        self,
        qtbot,
        sample_eeg_emg_for_viewer,
        sample_emg_filter,
        tmp_path,
    ):
        """Jump to next undefined epoch."""
        n_epochs = sample_eeg_emg_for_viewer["n_epochs"]
        labels = np.ones(n_epochs, dtype=int)
        labels[5] = UNDEFINED_LABEL
        labels[10] = UNDEFINED_LABEL

        label_file = str(tmp_path / "test_labels.csv")
        window = ManualScoringWindow(
            eeg=sample_eeg_emg_for_viewer["eeg"],
            emg=sample_eeg_emg_for_viewer["emg"],
            label_file=label_file,
            labels=labels.copy(),
            confidence_scores=None,
            sampling_rate=sample_eeg_emg_for_viewer["sampling_rate"],
            epoch_length=sample_eeg_emg_for_viewer["epoch_length"],
            emg_filter=sample_emg_filter,
        )
        qtbot.addWidget(window)
        window.show()

        assert window.epoch == 0
        window.jump_to_next_state("right", "undefined")
        assert window.epoch == 5

        window.jump_to_next_state("right", "undefined")
        assert window.epoch == 10
        sync_and_close(window)

    def test_jump_to_previous_different_state(self, manual_scoring_window):
        """Jump backward to an epoch with a different brain state."""
        # Labels start with [1, 1, 2, 2, 3, 3, ...]
        window = manual_scoring_window

        # Navigate to epoch 5 (label=3)
        for _ in range(5):
            window.shift_epoch("right")
        assert window.epoch == 5

        # Jump left to different state — should land on epoch 3 (last epoch with label=2)
        window.jump_to_next_state("left", "different")
        assert window.epoch == 3
        sync_and_close(window)

    def test_jump_with_no_match_does_not_move(
        self,
        qtbot,
        sample_eeg_emg_for_viewer,
        sample_emg_filter,
        tmp_path,
    ):
        """Jump with no matching epoch ahead stays at current position."""
        n_epochs = sample_eeg_emg_for_viewer["n_epochs"]
        labels = np.ones(n_epochs, dtype=int)

        label_file = str(tmp_path / "test_labels.csv")
        window = ManualScoringWindow(
            eeg=sample_eeg_emg_for_viewer["eeg"],
            emg=sample_eeg_emg_for_viewer["emg"],
            label_file=label_file,
            labels=labels.copy(),
            confidence_scores=None,
            sampling_rate=sample_eeg_emg_for_viewer["sampling_rate"],
            epoch_length=sample_eeg_emg_for_viewer["epoch_length"],
            emg_filter=sample_emg_filter,
        )
        qtbot.addWidget(window)
        window.show()

        assert window.epoch == 0
        window.jump_to_next_state("right", "different")
        assert window.epoch == 0

        window.jump_to_next_state("right", "undefined")
        assert window.epoch == 0
        sync_and_close(window)


@pytest.mark.gui
class TestManualScoringSave:
    """Tests for save functionality in ManualScoringWindow."""

    def test_save_updates_last_saved_labels(self, manual_scoring_window):
        """Saving updates last_saved_labels to match current labels."""
        window = manual_scoring_window

        window.modify_current_epoch_label(2)
        assert not np.array_equal(window.labels, window.last_saved_labels)

        window.save()
        assert np.array_equal(window.labels, window.last_saved_labels)
        window.close()

    def test_save_writes_to_file(self, manual_scoring_window):
        """Saving writes labels to the label file on disk."""
        from pathlib import Path

        import pandas as pd

        window = manual_scoring_window

        window.modify_current_epoch_label(2)
        window.save()

        label_file = Path(window.label_file)
        assert label_file.exists()
        saved = pd.read_csv(label_file)
        assert saved["brain_state"].iloc[0] == 2
        window.close()
