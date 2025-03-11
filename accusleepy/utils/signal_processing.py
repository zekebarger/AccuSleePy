import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import scipy.io
from scipy.signal import ShortTimeFFT, windows, butter, filtfilt

from accusleepy.utils.constants import (
    FILENAME_COL,
    LABEL_COL,
    EMG_COPIES,
    MIXTURE_WEIGHTS,
    N_CLASSES,
)

ABS_MAX_Z_SCORE = 3.5  # matlab version is 4.5


def load_files(file_path):
    eeg = scipy.io.loadmat(os.path.join(file_path, "EEG.mat"))["EEG"].T[0]
    emg = scipy.io.loadmat(os.path.join(file_path, "EMG.mat"))["EMG"].T[0]
    labels = scipy.io.loadmat(os.path.join(file_path, "labels.mat"))["labels"].T[0]
    return eeg, emg, labels


def truncate_signals(eeg, emg, sampling_rate, epoch_length):
    samples_per_epoch = int(sampling_rate * epoch_length)

    new_eeg = eeg[: len(eeg) - (len(eeg) % samples_per_epoch)]
    new_emg = emg[: len(emg) - (len(emg) % samples_per_epoch)]
    return new_eeg, new_emg


def create_spectrogram(eeg, sampling_rate, epoch_length):
    MIN_WINDOW_LEN = 5

    window_length_sec = max(MIN_WINDOW_LEN, epoch_length)
    window_length = int(window_length_sec * sampling_rate)

    # simplest possible window
    win = windows.boxcar(window_length)
    hop = int(epoch_length * sampling_rate)

    SFT = ShortTimeFFT(win=win, hop=hop, fs=sampling_rate, scale_to="psd")
    s = SFT.stft(eeg)
    f = SFT.f
    # TODO ok i really need to understand the details better,
    # but for now I'm just truncating the output
    # need to mess around with padding a bit

    return np.abs(s[:, :-1]), f


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


def create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length):
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
        [modified_spectrogram, np.matlib.repmat(emg_log_rms, EMG_COPIES, 1)]
    )
    return output


def mixture_z_score_img(img, labels):
    means = list()
    variances = list()

    for i in range(N_CLASSES):
        means.append(np.mean(img[:, labels == i + 1], axis=1))  # TODO label jank
        variances.append(np.var(img[:, labels == i + 1], axis=1))

    means = np.array(means)
    variances = np.array(variances)

    mixture_means = means.T @ MIXTURE_WEIGHTS
    mixture_sds = np.sqrt(
        variances.T @ MIXTURE_WEIGHTS
        + ((mixture_means - np.tile(mixture_means, (N_CLASSES, 1))) ** 2).T
        @ MIXTURE_WEIGHTS
    )

    img = ((img.T - mixture_means) / mixture_sds).T
    img = (img + ABS_MAX_Z_SCORE) / (2 * ABS_MAX_Z_SCORE)
    img = np.clip(img, 0, 1)

    return img


def create_images_from_rec(
    file_path, output_path, output_prefix, sampling_rate, epoch_length, img_width=9
):
    eeg, emg, labels = load_files(file_path)
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)

    img = mixture_z_score_img(img, labels)

    # pad left and right sides
    pad_width = int((img_width - 1) / 2)
    img = np.concatenate(
        [
            np.tile(img[:, 0], (pad_width, 1)).T,
            img,
            np.tile(img[:, -1], (pad_width, 1)).T,
        ],
        axis=1,
    )

    img = np.clip(img * 255, 0, 255)
    img = img.astype(np.uint8)

    fnames = []

    for i in range(img.shape[1] - img_width + 1):
        im = img[:, i : (i + img_width)]
        fname = f"{output_prefix}_{i}_{labels[i]}.png"
        fnames.append(fname)
        Image.fromarray(im).save(os.path.join(output_path, fname))

    label_df = pd.DataFrame({FILENAME_COL: fnames, LABEL_COL: labels}).to_csv(
        os.path.join(output_path, "labels.csv"),
        index=False,
    )

    print("finished generating images")
