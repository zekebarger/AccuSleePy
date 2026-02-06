"""Shared fixtures for AccuSleePy integration tests."""

import numpy as np
import pandas as pd
import pytest

from accusleepy.constants import BRAIN_STATE_COL, EEG_COL, EMG_COL
from accusleepy.fileio import Recording


@pytest.fixture
def epoch_length():
    """Epoch length to use in integration tests."""
    return 4


@pytest.fixture
def epochs_per_img():
    """Epochs per image to use for model training in integration tests."""
    return 9


@pytest.fixture
def calibration_fraction():
    """Calibration fraction for model training in integration tests."""
    return 0.2


@pytest.fixture
def synthetic_recording_data(epoch_length):
    """Generate synthetic EEG/EMG recording data.

    Creates 32 epochs at 128 Hz with 4-second epochs.
    Adds sinusoidal components to EEG for realistic spectrograms.
    """
    sampling_rate = 128
    n_epochs = 32
    n_samples = sampling_rate * epoch_length * n_epochs

    rng = np.random.default_rng(42)

    # Generate time vector
    t = np.arange(n_samples) / sampling_rate

    # Add sinusoidal components for realistic spectrograms
    # Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz)
    eeg = (
        rng.normal(0, 0.5, n_samples)
        + 0.5 * np.sin(2 * np.pi * 2 * t)  # Delta
        + 0.3 * np.sin(2 * np.pi * 6 * t)  # Theta
        + 0.2 * np.sin(2 * np.pi * 10 * t)  # Alpha
    )

    # EMG with some muscle activity variation
    emg = rng.normal(0, 1, n_samples) + 0.1 * np.sin(2 * np.pi * 30 * t)

    return {
        "eeg": eeg,
        "emg": emg,
        "sampling_rate": sampling_rate,
        "epoch_length": epoch_length,
        "n_epochs": n_epochs,
    }


@pytest.fixture
def synthetic_recording_file(tmp_path, synthetic_recording_data):
    """Create a CSV recording file on disk."""
    recording_file = tmp_path / "recording.csv"
    pd.DataFrame(
        {
            EEG_COL: synthetic_recording_data["eeg"],
            EMG_COL: synthetic_recording_data["emg"],
        }
    ).to_csv(recording_file, index=False)
    return recording_file


@pytest.fixture
def synthetic_label_file(tmp_path, synthetic_recording_data, sample_brain_state_set):
    """Create a CSV label file with balanced classes.

    Ensures each brain state has sufficient epochs for training.
    """
    n_epochs = synthetic_recording_data["n_epochs"]

    # Create balanced labels ensuring each class has enough samples
    rng = np.random.default_rng(42)
    scored_states = [s for s in sample_brain_state_set.brain_states if s.is_scored]
    state_digits = [s.digit for s in scored_states]

    # Distribute epochs evenly across states
    labels = np.array([state_digits[i % len(state_digits)] for i in range(n_epochs)])
    rng.shuffle(labels)

    label_file = tmp_path / "labels.csv"
    pd.DataFrame({BRAIN_STATE_COL: labels}).to_csv(label_file, index=False)
    return label_file


@pytest.fixture
def synthetic_recording(
    synthetic_recording_file,
    synthetic_label_file,
    synthetic_recording_data,
    tmp_path,
):
    """Complete Recording object ready for services."""
    return Recording(
        name=1,
        recording_file=str(synthetic_recording_file),
        label_file=str(synthetic_label_file),
        calibration_file=str(tmp_path / "calibration.csv"),
        sampling_rate=synthetic_recording_data["sampling_rate"],
    )
