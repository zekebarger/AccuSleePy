"""Shared fixtures for AccuSleePy GUI tests"""

import os

# Use offscreen rendering to avoid visible windows during tests
# Must be set before Qt is imported
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from accusleepy.constants import UNDEFINED_LABEL


@pytest.fixture
def sample_eeg_emg_for_viewer():
    """Generate synthetic EEG/EMG data suitable for the manual scoring window.

    Creates data for 100 epochs at 128 Hz with 4-second epochs.
    """
    sampling_rate = 128
    epoch_length = 4
    n_epochs = 100
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
    return labels.astype(int)
