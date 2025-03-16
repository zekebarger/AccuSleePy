import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import butter, filtfilt

import accusleepy.utils.constants as c
from accusleepy.utils.fileio import load_labels, load_recording
from accusleepy.utils.misc import Recording
from accusleepy.utils.multitaper import spectrogram

ABS_MAX_Z_SCORE = 3.5  # matlab version is 4.5
SPECTROGRAM_UPPER_FREQ = 64


def truncate_signals(
    eeg: np.array, emg: np.array, sampling_rate: int | float, epoch_length: int | float
) -> (np.array, np.array):
    samples_per_epoch = int(sampling_rate * epoch_length)

    new_eeg = eeg[: len(eeg) - (len(eeg) % samples_per_epoch)]
    new_emg = emg[: len(emg) - (len(emg) % samples_per_epoch)]
    return new_eeg, new_emg


def create_spectrogram(
    eeg: np.array,
    sampling_rate: int | float,
    epoch_length: int | float,
    time_bandwidth=2,
    n_tapers=3,
) -> (np.array, np.array):
    window_length_sec = max(c.MIN_WINDOW_LEN, epoch_length)
    pad_length = int((sampling_rate * (window_length_sec - epoch_length) / 2))
    padded_eeg = np.concatenate(
        [eeg[:pad_length][::-1], eeg, eeg[(len(eeg) - pad_length) :][::-1]]
    )

    spec, _, f = spectrogram(
        padded_eeg,
        sampling_rate,
        frequency_range=[0, SPECTROGRAM_UPPER_FREQ],
        time_bandwidth=time_bandwidth,
        num_tapers=n_tapers,
        window_params=[window_length_sec, epoch_length],
        min_nfft=0,
        detrend_opt="off",
        multiprocess=True,
        plot_on=False,
        return_fig=False,
        verbose=False,
    )

    # resample frequencies for consistency
    target_frequencies = np.arange(0, SPECTROGRAM_UPPER_FREQ, 1 / c.MIN_WINDOW_LEN)
    freq_idx = list()
    for i in target_frequencies:
        freq_idx.append(np.argmin(np.abs(f - i)))
    f = f[freq_idx]
    spec = spec[freq_idx, :]

    return spec, f


def process_emg(
    emg: np.array, sampling_rate: int | float, epoch_length: int | float
) -> np.array:
    order = 8
    bp_lower = 20
    bp_upper = 50

    b, a = butter(
        N=order,
        Wn=[bp_lower, bp_upper],
        btype="bandpass",
        output="ba",
        fs=sampling_rate,
    )
    filtered = filtfilt(b, a, x=emg, padlen=sampling_rate)  # padlen?

    samples_per_epoch = int(sampling_rate * epoch_length)
    reshaped = np.reshape(
        filtered,
        [int(len(emg) / samples_per_epoch), samples_per_epoch],
    )
    rms = np.sqrt(np.mean(np.power(reshaped, 2), axis=1))

    return np.log(rms)


def create_eeg_emg_image(
    eeg: np.array,
    emg: np.array,
    sampling_rate: int | float,
    epoch_length: int | float,
    emg_copies: int = c.EMG_COPIES,
) -> np.array:
    spec, f = create_spectrogram(eeg, sampling_rate, epoch_length)
    f_lower_idx = sum(f < c.DOWNSAMPLING_START_FREQ)
    f_upper_idx = sum(f < c.UPPER_FREQ)

    modified_spectrogram = np.log(
        spec[
            np.concatenate(
                [np.arange(0, f_lower_idx), np.arange(f_lower_idx, f_upper_idx, 2)]
            ),
            :,
        ]
    )

    emg_log_rms = process_emg(emg, sampling_rate, epoch_length)
    output = np.concatenate(
        [modified_spectrogram, np.tile(emg_log_rms, (emg_copies, 1))]
    )
    return output


def get_mixture_values(img: np.array, labels: np.array) -> (np.array, np.array):
    # labels = CLASSES

    means = list()
    variances = list()

    for i in range(c.BRAIN_STATE_MAPPER.n_classes):
        means.append(np.mean(img[:, labels == i], axis=1))
        variances.append(np.var(img[:, labels == i], axis=1))

    means = np.array(means)
    variances = np.array(variances)

    mixture_means = means.T @ c.MIXTURE_WEIGHTS
    mixture_sds = np.sqrt(
        variances.T @ c.MIXTURE_WEIGHTS
        + (
            (
                mixture_means
                - np.tile(mixture_means, (c.BRAIN_STATE_MAPPER.n_classes, 1))
            )
            ** 2
        ).T
        @ c.MIXTURE_WEIGHTS
    )

    return mixture_means, mixture_sds


def mixture_z_score_img(
    img: np.array,
    labels: np.array = None,
    mixture_means: np.array = None,
    mixture_sds: np.array = None,
) -> np.array:
    # labels = CLASSES

    if labels is None and (mixture_means is None or mixture_sds is None):
        raise Exception("must provide either labels or mixture means+SDs")
    if labels is not None and ((mixture_means is not None) ^ (mixture_sds is not None)):
        warnings.warn("labels were given, mixture means / SDs will be ignored")

    ABS_MAX_Z_SCORE = 3.5  # matlab version is 4.5

    if labels is not None:
        mixture_means, mixture_sds = get_mixture_values(img, labels)

    img = ((img.T - mixture_means) / mixture_sds).T
    img = (img + ABS_MAX_Z_SCORE) / (2 * ABS_MAX_Z_SCORE)
    img = np.clip(img, 0, 1)

    return img


def format_img(img: np.array, epochs_per_img: int) -> np.array:
    # pad left and right sides
    pad_width = int((epochs_per_img - 1) / 2)
    img = np.concatenate(
        [
            np.tile(img[:, 0], (pad_width, 1)).T,
            img,
            np.tile(img[:, -1], (pad_width, 1)).T,
        ],
        axis=1,
    )

    # use 8-bit values
    img = np.clip(img * 255, 0, 255)
    img = img.astype(np.uint8)

    return img


def create_training_images(
    recordings: list[Recording],
    output_path: str,
    epoch_length: int | float,
    epochs_per_img: int = c.EPOCHS_PER_IMG,
) -> None:
    n_files = len(recordings)

    filenames = list()
    all_labels = np.empty(0).astype(int)
    for i in range(n_files):
        eeg, emg = load_recording(recordings[i].recording_file)
        labels = load_labels(recordings[i].label_file)

        labels = c.BRAIN_STATE_MAPPER.convert_digit_to_class(labels)
        img = create_eeg_emg_image(eeg, emg, recordings[i].sampling_rate, epoch_length)
        img = mixture_z_score_img(img, labels)
        img = format_img(img, epochs_per_img)

        for j in range(img.shape[1] - epochs_per_img + 1):
            im = img[:, j : (j + epochs_per_img)]
            filename = f"file_{i}_{j}_{labels[j]}.png"
            filenames.append(filename)
            Image.fromarray(im).save(os.path.join(output_path, filename))

        all_labels = np.concatenate([all_labels, labels])

    pd.DataFrame({c.FILENAME_COL: filenames, c.LABEL_COL: all_labels}).to_csv(
        os.path.join(output_path, "labels.csv"),
        index=False,
    )

    print(f"finished generating {len(all_labels)} images")
