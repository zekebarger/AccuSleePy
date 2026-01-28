import numpy as np
import pytest

from accusleepy.constants import (
    MIN_WINDOW_LEN,
    SPECTROGRAM_UPPER_FREQ,
)
from accusleepy.signal_processing import (
    create_eeg_emg_image,
    create_spectrogram,
    format_img,
    get_emg_power,
    get_mixture_values,
    mixture_z_score_img,
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


def test_create_eeg_emg_image(sample_emg_filter):
    """Test that this function produces some output"""

    sampling_rate = 128
    epoch_length = 4
    n_epochs = 100
    n_samples = sampling_rate * epoch_length * n_epochs

    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 1, n_samples)
    emg = rng.normal(0, 1, n_samples)

    img = create_eeg_emg_image(
        eeg=eeg,
        emg=emg,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        emg_filter=sample_emg_filter,
    )

    assert type(img) is np.ndarray
    assert img.ndim == 2


def test_get_emg_power(sample_emg_filter):
    """EMG power calculation produces output"""
    sampling_rate = 128
    epoch_length = 4
    n_epochs = 30
    n_samples = sampling_rate * epoch_length * n_epochs

    rng = np.random.default_rng(42)
    emg = rng.normal(0, 1, n_samples)

    emg_power = get_emg_power(
        emg=emg,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        emg_filter=sample_emg_filter,
    )

    assert len(emg_power) == n_epochs
    assert emg_power.dtype == np.float64


def test_get_mixture_values(sample_brain_state_set):
    """Mixture values have expected shape"""
    n_features = 30
    n_epochs = 100

    rng = np.random.default_rng(42)
    img = rng.normal(0, 1, (n_features, n_epochs))

    # Create labels with all classes represented
    labels = np.array([i % sample_brain_state_set.n_classes for i in range(n_epochs)])

    mixture_means, mixture_sds = get_mixture_values(
        img=img, labels=labels, brain_state_set=sample_brain_state_set
    )

    # Should have one mean/SD per feature
    assert mixture_means.shape == (n_features,)
    assert mixture_sds.shape == (n_features,)


def test_mixture_z_score_img(sample_brain_state_set):
    """Z-scoring produces normalized output"""
    n_features = 30
    n_epochs = 100

    rng = np.random.default_rng(42)
    img = rng.normal(0, 1, (n_features, n_epochs)) * 255

    # Create labels with all classes represented
    labels = np.array([i % sample_brain_state_set.n_classes for i in range(n_epochs)])

    z_scored_img, had_zero_variance = mixture_z_score_img(
        img=img, brain_state_set=sample_brain_state_set, labels=labels
    )

    # Output should be clipped to [0, 1]
    assert z_scored_img.min() >= 0
    assert z_scored_img.max() <= 1
    # Should have same shape as input
    assert z_scored_img.shape == img.shape
    # Normal random data shouldn't have zero variance features
    assert not had_zero_variance


def test_mixture_z_score_img_with_means_sds(sample_brain_state_set):
    """Z-scoring works with provided means/SDs"""
    n_features = 30
    n_epochs = 100

    rng = np.random.default_rng(42)
    img = rng.normal(0, 1, (n_features, n_epochs)) * 255

    # Provide custom means and SDs
    mixture_means = np.zeros(n_features)
    mixture_sds = np.ones(n_features)

    z_scored_img, had_zero_variance = mixture_z_score_img(
        img=img,
        brain_state_set=sample_brain_state_set,
        mixture_means=mixture_means,
        mixture_sds=mixture_sds,
    )

    # Output should be clipped to [0, 1]
    assert z_scored_img.min() >= 0
    assert z_scored_img.max() <= 1
    assert z_scored_img.shape == img.shape


def test_mixture_z_score_img_requires_labels_or_means(sample_brain_state_set):
    """Raises error when neither labels nor means/SDs provided"""
    img = np.random.randn(10, 20)

    with pytest.raises(ValueError, match="must provide either labels or mixture"):
        mixture_z_score_img(img=img, brain_state_set=sample_brain_state_set)


def test_format_img_with_padding():
    """format_img produces output"""
    n_features = 30
    n_epochs = 100
    epochs_per_img = 9

    rng = np.random.default_rng(42)
    img = rng.uniform(0, 1, (n_features, n_epochs))

    formatted = format_img(img=img, epochs_per_img=epochs_per_img, add_padding=True)

    assert formatted.dtype == np.uint8
    assert formatted.min() >= 0
    assert formatted.max() <= 255


def test_format_img_without_padding():
    """No padding when add_padding=False"""
    n_features = 30
    n_epochs = 100
    epochs_per_img = 9

    rng = np.random.default_rng(42)
    img = rng.uniform(0, 1, (n_features, n_epochs))

    formatted = format_img(img=img, epochs_per_img=epochs_per_img, add_padding=False)

    # Should have same width as input (no padding added)
    assert formatted.shape == (n_features, n_epochs)
    assert formatted.dtype == np.uint8
