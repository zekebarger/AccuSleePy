"""Shared fixtures for AccuSleePy GUI tests"""

import os

# Use offscreen rendering to avoid visible windows during tests
# Must be set before Qt is imported
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from accusleepy.constants import UNDEFINED_LABEL


@pytest.fixture
def epoch_length():
    """Epoch length to use for GUI tests."""
    return 4


@pytest.fixture
def sampling_rate():
    """Sampling rate to use for GUI tests."""
    return 128


@pytest.fixture
def sample_eeg_emg_for_viewer(epoch_length, sampling_rate):
    """Generate synthetic EEG/EMG data suitable for the manual scoring window.

    Creates data for 24 epochs at 128 Hz with 4-second epochs.
    """
    n_epochs = 24
    n_samples = sampling_rate * epoch_length * n_epochs

    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 1, n_samples)
    emg = rng.normal(0, 1, n_samples)

    return {
        "eeg": eeg,
        "emg": emg,
        "sampling_rate": sampling_rate,
        "epoch_length": epoch_length,
        "n_epochs": n_epochs,
    }


@pytest.fixture
def sample_labels_for_viewer(sample_eeg_emg_for_viewer):
    """Generate sample labels for the manual scoring window.

    Labels use digits matching config.json: 1 (REM), 2 (Wake), 3 (NREM), -1 (undefined)
    """
    n_epochs = sample_eeg_emg_for_viewer["n_epochs"]
    # Mix of labeled and undefined epochs using config's digit values
    rng = np.random.default_rng(42)
    labels = rng.choice([UNDEFINED_LABEL, 1, 2, 3], size=n_epochs)
    # Ensure the first few labels are always the same
    labels[0:6] = [1, 1, 2, 2, 3, 3]
    return labels.astype(int)


@pytest.fixture
def manual_scoring_window(
    qtbot,
    sample_eeg_emg_for_viewer,
    sample_labels_for_viewer,
    sample_emg_filter,
    tmp_path,
):
    """Create a ManualScoringWindow with standard test data.

    If a test modifies the labels, it should call
    ``sync_and_close(window)`` at the end to prevent the
    "unsaved changes" dialog from blocking in offscreen mode.
    Access the label file path via ``window.label_file``.
    """
    from accusleepy.gui.manual_scoring import ManualScoringWindow

    label_file = str(tmp_path / "test_labels.csv")
    window = ManualScoringWindow(
        eeg=sample_eeg_emg_for_viewer["eeg"],
        emg=sample_eeg_emg_for_viewer["emg"],
        label_file=label_file,
        labels=sample_labels_for_viewer.copy(),
        confidence_scores=None,
        sampling_rate=sample_eeg_emg_for_viewer["sampling_rate"],
        epoch_length=sample_eeg_emg_for_viewer["epoch_length"],
        emg_filter=sample_emg_filter,
    )
    qtbot.addWidget(window)
    window.show()
    return window


@pytest.fixture
def synthetic_recording_file_for_gui(tmp_path, sample_eeg_emg_for_viewer):
    """Create a CSV recording file for GUI tests."""
    recording_file = tmp_path / "recording.csv"
    pd.DataFrame(
        {
            "eeg": sample_eeg_emg_for_viewer["eeg"],
            "emg": sample_eeg_emg_for_viewer["emg"],
        }
    ).to_csv(recording_file, index=False)
    return recording_file


@pytest.fixture
def synthetic_label_file_for_gui(tmp_path, sample_labels_for_viewer):
    """Create a CSV label file for GUI tests."""
    label_file = tmp_path / "labels.csv"
    pd.DataFrame({"brain_state": sample_labels_for_viewer}).to_csv(
        label_file, index=False
    )
    return label_file
