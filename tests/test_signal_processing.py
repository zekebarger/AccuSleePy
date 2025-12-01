import numpy as np
import pytest

from accusleepy.constants import (
    MIN_WINDOW_LEN,
    SPECTROGRAM_UPPER_FREQ,
)
from accusleepy.fileio import EMGFilter
from accusleepy.signal_processing import (
    create_eeg_emg_image,
    create_spectrogram,
    resample,
    standardize_signal_length,
)


def test_create_spectrogram():
    """Spectrogram is created with correct shape and type"""
    sampling_rate = 512
    n_samples = sampling_rate * 60
    eeg = np.sin(10 * np.arange(n_samples) / sampling_rate)
    epoch_length = 5
    spectrogram, f = create_spectrogram(
        eeg=eeg, sampling_rate=sampling_rate, epoch_length=epoch_length
    )
    assert type(spectrogram) is np.ndarray
    assert spectrogram.shape == (
        len(np.arange(0, SPECTROGRAM_UPPER_FREQ, 1 / MIN_WINDOW_LEN)),
        n_samples / (sampling_rate * epoch_length),
    )


def test_resample_no_change():
    """Resample function makes no unnecessary changes"""
    eeg = np.zeros(1000)
    emg = np.zeros(1000)
    sampling_rate = 10
    epoch_length = 10
    new_eeg, new_emg, new_sampling_rate = resample(
        eeg=eeg, emg=emg, sampling_rate=sampling_rate, epoch_length=epoch_length
    )
    assert new_sampling_rate == sampling_rate
    assert len(eeg) == len(new_eeg)


def test_resample_with_change():
    """New sampling rate is correct"""
    eeg = np.zeros(1000)
    emg = np.zeros(1000)
    sampling_rate = 12.5
    epoch_length = 5
    new_eeg, new_emg, new_sampling_rate = resample(
        eeg=eeg, emg=emg, sampling_rate=sampling_rate, epoch_length=epoch_length
    )
    assert new_sampling_rate * epoch_length % 1 == pytest.approx(0)
    assert len(eeg) / sampling_rate == pytest.approx(len(new_eeg) / new_sampling_rate)


def test_standardize_length_no_change():
    """Standardizing length makes no unnecessary changes"""
    eeg = np.zeros(1000)
    emg = np.zeros(1000)
    sampling_rate = 10
    epoch_length = 5
    new_eeg, new_emg = standardize_signal_length(
        eeg=eeg, emg=emg, sampling_rate=sampling_rate, epoch_length=epoch_length
    )
    assert len(new_eeg) == len(eeg)


def test_standardize_length():
    """Standardizing length works correctly"""
    sampling_rate = 10
    epoch_length = 5
    base_epoch_count = 20

    samples_per_epoch = round(sampling_rate * epoch_length)
    base_length = samples_per_epoch * base_epoch_count

    test_lengths = [
        base_length - round(samples_per_epoch / 2) - 1,
        base_length - round(samples_per_epoch / 2),
        base_length - round(samples_per_epoch / 2) + 1,
        base_length,
        base_length + round(samples_per_epoch / 2) - 1,
        base_length + round(samples_per_epoch / 2),
        base_length + round(samples_per_epoch / 2) + 1,
    ]
    target_lengths = [
        round(i * samples_per_epoch)
        for i in [
            base_epoch_count - 1,
            base_epoch_count,
            base_epoch_count,
            base_epoch_count,
            base_epoch_count,
            base_epoch_count + 1,
            base_epoch_count + 1,
        ]
    ]

    for i, test_length in enumerate(test_lengths):
        eeg = np.zeros(test_length)
        emg = np.zeros(test_length)
        new_eeg, new_emg = standardize_signal_length(
            eeg=eeg, emg=emg, sampling_rate=sampling_rate, epoch_length=epoch_length
        )
        assert len(new_eeg) == target_lengths[i]


def test_create_eeg_emg_image():
    """Test that this function produces some output"""

    sampling_rate = 128
    epoch_length = 4
    n_epochs = 100
    n_samples = sampling_rate * epoch_length * n_epochs

    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 1, n_samples)
    emg = rng.normal(0, 1, n_samples)

    emg_filter = EMGFilter(
        order=8,
        bp_lower=20,
        bp_upper=50,
    )

    img = create_eeg_emg_image(
        eeg=eeg,
        emg=emg,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        emg_filter=emg_filter,
    )

    assert type(img) is np.ndarray
    assert img.ndim == 2
