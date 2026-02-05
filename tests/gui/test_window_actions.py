"""GUI Tests: Window Instantiation & Basic Lifecycle

These tests verify that windows open and close without crashes.
"""

import numpy as np
import pytest

from accusleepy.gui.main import AccuSleepWindow


@pytest.mark.gui
class TestMainWindowLifecycle:
    """Tests for AccuSleepWindow instantiation and lifecycle"""

    def test_main_window_opens_and_closes(self, qtbot):
        """AccuSleepWindow instantiates and shows without errors"""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        assert window.isVisible()
        assert window.ui is not None
        # check for correct default state
        assert window.recording_manager is not None
        assert window.config is not None
        assert window.recording_manager.current is not None
        assert window.loaded_model.model is None
        window.close()
        assert not window.isVisible()


@pytest.mark.gui
class TestManualScoringWindowLifecycle:
    """Tests for ManualScoringWindow instantiation and lifecycle"""

    def test_manual_scoring_window_opens(
        self, manual_scoring_window, sample_eeg_emg_for_viewer
    ):
        """ManualScoringWindow opens with valid data"""
        assert manual_scoring_window.isVisible()
        assert manual_scoring_window.n_epochs == sample_eeg_emg_for_viewer["n_epochs"]
        assert (
            manual_scoring_window.sampling_rate
            == sample_eeg_emg_for_viewer["sampling_rate"]
        )
        manual_scoring_window.close()

    def test_manual_scoring_window_closes_without_changes(self, manual_scoring_window):
        """ManualScoringWindow closes cleanly when no changes are made"""
        assert manual_scoring_window.isVisible()

        # Perform an action that doesn't change the labels
        manual_scoring_window.modify_current_epoch_label(
            manual_scoring_window.labels[0]
        )
        assert np.array_equal(
            manual_scoring_window.labels, manual_scoring_window.last_saved_labels
        )

        manual_scoring_window.close()
        assert not manual_scoring_window.isVisible()
