import numpy as np
import pytest

from accusleepy.signal_processing import (
    enforce_min_bout_length,
    resample,
    standardize_signal_length,
)


def test_resample_no_change():
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
    eeg = np.zeros(1000)
    emg = np.zeros(1000)
    sampling_rate = 10
    epoch_length = 5
    new_eeg, new_emg = standardize_signal_length(
        eeg=eeg, emg=emg, sampling_rate=sampling_rate, epoch_length=epoch_length
    )
    assert len(new_eeg) == len(eeg)


def test_standardize_length():
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


def test_basic_bout_fix():
    min_bout_length = 2
    epoch_length = 1
    labels = np.array([1, 1, 2, 2, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3])
    target = np.array([1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)


def test_even_number_bout_sequence():
    min_bout_length = 3
    epoch_length = 1
    labels = np.array([1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    target = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)


def test_odd_number_bout_sequence():
    min_bout_length = 3
    epoch_length = 1
    labels = np.array([1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1])
    target = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)
