"""Phase 1 GUI Tests: Window Instantiation & Basic Lifecycle

These tests verify that windows open and close without crashes.
"""

import numpy as np
import pytest

from accusleepy.gui.main import AccuSleepWindow
from accusleepy.gui.manual_scoring import ManualScoringWindow


@pytest.mark.gui
class TestMainWindowLifecycle:
    """Tests for AccuSleepWindow instantiation and lifecycle"""

    def test_main_window_opens(self, qtbot):
        """AccuSleepWindow instantiates and shows without errors"""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        # Window should be visible (show() is called in __init__)
        assert window.isVisible()

        # Key UI components should exist
        assert window.ui is not None
        assert window.recording_manager is not None
        assert window.config is not None

    def test_main_window_closes(self, qtbot):
        """AccuSleepWindow closes cleanly"""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        assert window.isVisible()

        # Close the window
        window.close()

        assert not window.isVisible()

    def test_main_window_initial_state(self, qtbot):
        """Main window initializes with correct default state"""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        # Should have one recording by default (from RecordingListManager)
        assert window.recording_manager.current is not None

        # Loaded model should be empty initially
        assert window.loaded_model.model is None


@pytest.mark.gui
class TestManualScoringWindowLifecycle:
    """Tests for ManualScoringWindow instantiation and lifecycle"""

    def test_manual_scoring_window_opens(
        self,
        qtbot,
        sample_eeg_emg_for_viewer,
        sample_labels_for_viewer,
        sample_emg_filter,
        tmp_path,
    ):
        """ManualScoringWindow opens with valid data"""
        label_file = str(tmp_path / "test_labels.csv")

        window = ManualScoringWindow(
            eeg=sample_eeg_emg_for_viewer["eeg"],
            emg=sample_eeg_emg_for_viewer["emg"],
            label_file=label_file,
            labels=sample_labels_for_viewer,
            confidence_scores=None,
            sampling_rate=sample_eeg_emg_for_viewer["sampling_rate"],
            epoch_length=sample_eeg_emg_for_viewer["epoch_length"],
            emg_filter=sample_emg_filter,
        )
        qtbot.addWidget(window)

        window.show()
        assert window.isVisible()

        # Check that key attributes are set
        assert window.n_epochs == sample_eeg_emg_for_viewer["n_epochs"]
        assert window.sampling_rate == sample_eeg_emg_for_viewer["sampling_rate"]

    def test_manual_scoring_window_closes_without_changes(
        self,
        qtbot,
        sample_eeg_emg_for_viewer,
        sample_labels_for_viewer,
        sample_emg_filter,
        tmp_path,
    ):
        """ManualScoringWindow closes cleanly when no changes made"""
        label_file = str(tmp_path / "test_labels.csv")

        window = ManualScoringWindow(
            eeg=sample_eeg_emg_for_viewer["eeg"],
            emg=sample_eeg_emg_for_viewer["emg"],
            label_file=label_file,
            labels=sample_labels_for_viewer,
            confidence_scores=None,
            sampling_rate=sample_eeg_emg_for_viewer["sampling_rate"],
            epoch_length=sample_eeg_emg_for_viewer["epoch_length"],
            emg_filter=sample_emg_filter,
        )
        qtbot.addWidget(window)

        window.show()
        assert window.isVisible()

        # No changes made, labels should match last_saved_labels
        assert np.array_equal(window.labels, window.last_saved_labels)

        # Close should succeed without dialog
        window.close()
        assert not window.isVisible()
