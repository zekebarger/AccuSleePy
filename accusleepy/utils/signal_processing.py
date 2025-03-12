import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import butter, filtfilt

import accusleepy.utils.constants as c
from accusleepy.utils.multitaper import spectrogram

ABS_MAX_Z_SCORE = 3.5  # matlab version is 4.5


def truncate_signals(eeg, emg, sampling_rate, epoch_length):
    samples_per_epoch = int(sampling_rate * epoch_length)

    new_eeg = eeg[: len(eeg) - (len(eeg) % samples_per_epoch)]
    new_emg = emg[: len(emg) - (len(emg) % samples_per_epoch)]
    return new_eeg, new_emg


def create_spectrogram(eeg, sampling_rate, epoch_length):
    MIN_WINDOW_LEN = 5

    window_length_sec = max(MIN_WINDOW_LEN, epoch_length)
    pad_length = int((sampling_rate * (window_length_sec - epoch_length) / 2))
    padded_eeg = np.concatenate([eeg[:pad_length][::-1], eeg, eeg[-pad_length:][::-1]])

    spec, _, f = spectrogram(
        padded_eeg,
        sampling_rate,
        frequency_range=[0, 64],
        time_bandwidth=5,
        num_tapers=3,
        window_params=[window_length_sec, epoch_length],
        min_nfft=0,
        detrend_opt="off",
        multiprocess=True,
        plot_on=False,
        return_fig=False,
        verbose=False,
    )

    return spec, f


def process_emg(emg, sampling_rate, epoch_length):
    ORDER = 8
    BP_LOWER = 20
    BP_UPPER = 50

    b, a = butter(
        N=ORDER,
        Wn=[BP_LOWER, BP_UPPER],
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
    eeg, emg, sampling_rate, epoch_length, emg_copies=c.EMG_COPIES
):
    DOWNSAMPLING_START_FREQ = 20
    UPPER_FREQ = 50

    spectrogram, f = create_spectrogram(eeg, sampling_rate, epoch_length)
    f_lower_idx = sum(f < DOWNSAMPLING_START_FREQ)
    f_upper_idx = sum(f < UPPER_FREQ)

    modified_spectrogram = np.log(
        spectrogram[
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


def get_mixture_values(img, labels):
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


def mixture_z_score_img(img, labels=None, mixture_means=None, mixture_sds=None):
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


def format_img(img, epochs_per_img):
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
    eeg,
    emg,
    labels,
    output_path,
    output_prefix,
    sampling_rate,
    epoch_length,
    epochs_per_img=c.EPOCHS_PER_IMG,
):
    labels = c.BRAIN_STATE_MAPPER.convert_digit_to_class(labels)
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)
    img = mixture_z_score_img(img, labels)
    img = format_img(img, epochs_per_img)

    filenames = []

    for i in range(img.shape[1] - epochs_per_img + 1):
        im = img[:, i : (i + epochs_per_img)]
        filename = f"{output_prefix}_{i}_{labels[i]}.png"
        filenames.append(filename)
        Image.fromarray(im).save(os.path.join(output_path, filename))

    pd.DataFrame({c.FILENAME_COL: filenames, c.LABEL_COL: labels}).to_csv(
        os.path.join(output_path, "labels.csv"),
        index=False,
    )

    print("finished generating images")
