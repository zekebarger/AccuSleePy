import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange

from accusleepy.brain_state_set import BrainStateSet
from accusleepy.constants import (
    ANNOTATIONS_FILENAME,
    DEFAULT_MODEL_TYPE,
    DOWNSAMPLING_START_FREQ,
    EMG_COPIES,
    FILENAME_COL,
    LABEL_COL,
    MIN_WINDOW_LEN,
    UPPER_FREQ,
)
from accusleepy.fileio import Recording, load_labels, load_recording
from accusleepy.multitaper import spectrogram

# note: scipy is lazily imported

# clip mixture z-scores above and below this level
# in the matlab implementation, I used 4.5
ABS_MAX_Z_SCORE = 3.5
# upper frequency limit when generating EEG spectrograms
SPECTROGRAM_UPPER_FREQ = 64


def resample(
    eeg: np.array, emg: np.array, sampling_rate: int | float, epoch_length: int | float
) -> (np.array, np.array, float):
    """Resample recording so that epochs contain equal numbers of samples

    If the number of samples per epoch is not an integer, epoch-level calculations
    are much more difficult. To avoid this, we can resample the EEG and EMG signals
    and adjust the sampling rate accordingly.

    :param eeg: EEG signal
    :param emg: EMG signal
    :param sampling_rate: original sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :return: resampled EEG & EMG and updated sampling rate
    """
    samples_per_epoch = sampling_rate * epoch_length
    if samples_per_epoch % 1 == 0:
        return eeg, emg, sampling_rate

    resampled = list()
    for arr in [eeg, emg]:
        x = np.arange(0, arr.size)
        x_new = np.linspace(
            0,
            arr.size - 1,
            round(arr.size * np.ceil(samples_per_epoch) / samples_per_epoch),
        )
        resampled.append(np.interp(x_new, x, arr))

    eeg = resampled[0]
    emg = resampled[1]
    new_sampling_rate = np.ceil(samples_per_epoch) / samples_per_epoch * sampling_rate
    return eeg, emg, new_sampling_rate


def standardize_signal_length(
    eeg: np.array, emg: np.array, sampling_rate: int | float, epoch_length: int | float
) -> (np.array, np.array):
    """Truncate or pad EEG/EMG signals to have an integer number of epochs

    :param eeg: EEG signal
    :param emg: EMG signal
    :param sampling_rate: original sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :return: EEG and EMG signals
    """
    # since resample() was called, this will be extremely close to an integer
    samples_per_epoch = round(sampling_rate * epoch_length)

    # pad the signal at the end in case we need more samples
    eeg = np.concatenate((eeg, np.ones(samples_per_epoch) * eeg[-1]))
    emg = np.concatenate((emg, np.ones(samples_per_epoch) * emg[-1]))
    padded_signal_length = eeg.size

    # count samples that don't fit in any epoch
    excess_samples = padded_signal_length % samples_per_epoch
    # we will definitely remove those
    last_index = padded_signal_length - excess_samples
    # and if the last epoch of real data had more than half of
    # its samples missing, delete it
    if excess_samples < samples_per_epoch / 2:
        last_index -= samples_per_epoch

    return eeg[:last_index], emg[:last_index]


def resample_and_standardize(
    eeg: np.array, emg: np.array, sampling_rate: int | float, epoch_length: int | float
) -> (np.array, np.array, float):
    """Preprocess EEG and EMG signals

    Adjust the length and sampling rate of the EEG and EMG signals so that
    each epoch contains an integer number of samples and each recording
    contains an integer number of epochs.

    :param eeg: EEG signal
    :param emg: EMG signal
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :return: processed EEG & EMG signals, and the new sampling rate
    """
    eeg, emg, sampling_rate = resample(
        eeg=eeg, emg=emg, sampling_rate=sampling_rate, epoch_length=epoch_length
    )
    eeg, emg = standardize_signal_length(
        eeg=eeg, emg=emg, sampling_rate=sampling_rate, epoch_length=epoch_length
    )
    return eeg, emg, sampling_rate


def create_spectrogram(
    eeg: np.array,
    sampling_rate: int | float,
    epoch_length: int | float,
    time_bandwidth=2,
    n_tapers=3,
) -> (np.array, np.array):
    """Create an EEG spectrogram image

    :param eeg: EEG signal
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :param time_bandwidth: time-half bandwidth product
    :param n_tapers: number of DPSS tapers to use
    :return: spectrogram and its frequency axis
    """
    window_length_sec = max(MIN_WINDOW_LEN, epoch_length)
    # pad the EEG signal so that the first spectrogram window is centered
    # on the first epoch
    # it's possible there's some jank here, if this isn't close to an integer
    pad_length = round((sampling_rate * (window_length_sec - epoch_length) / 2))
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
    target_frequencies = np.arange(0, SPECTROGRAM_UPPER_FREQ, 1 / MIN_WINDOW_LEN)
    freq_idx = list()
    for i in target_frequencies:
        freq_idx.append(np.argmin(np.abs(f - i)))
    f = f[freq_idx]
    spec = spec[freq_idx, :]

    return spec, f


def get_emg_power(
    emg: np.array, sampling_rate: int | float, epoch_length: int | float
) -> np.array:
    """Calculate EMG power for each epoch

    This applies a 20-50 Hz bandpass filter to the EMG,  calculates the RMS
    in each epoch, and takes the log of the result.

    :param emg: EMG signal
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :return: EMG "power" for each epoch
    """
    from scipy.signal import butter, filtfilt

    # filter parameters
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
    filtered = filtfilt(b, a, x=emg, padlen=int(np.ceil(sampling_rate)))

    # since resample() was called, this will be extremely close to an integer
    samples_per_epoch = round(sampling_rate * epoch_length)
    reshaped = np.reshape(
        filtered,
        [round(len(emg) / samples_per_epoch), samples_per_epoch],
    )
    rms = np.sqrt(np.mean(np.power(reshaped, 2), axis=1))

    return np.log(rms)


def create_eeg_emg_image(
    eeg: np.array,
    emg: np.array,
    sampling_rate: int | float,
    epoch_length: int | float,
) -> np.array:
    """Stack EEG spectrogram and EMG power into an image

    This assumes that each epoch contains an integer number of samples and
    each recording contains an integer number of epochs. Note that a log
    transformation is applied to the spectrogram.

    :param eeg: EEG signal
    :param emg: EMG signal
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :return: combined EEG + EMG image for a recording
    """
    spec, f = create_spectrogram(eeg, sampling_rate, epoch_length)
    f_lower_idx = sum(f < DOWNSAMPLING_START_FREQ)
    f_upper_idx = sum(f < UPPER_FREQ)

    modified_spectrogram = np.log(
        spec[
            np.concatenate(
                [np.arange(0, f_lower_idx), np.arange(f_lower_idx, f_upper_idx, 2)]
            ),
            :,
        ]
    )

    emg_log_rms = get_emg_power(emg, sampling_rate, epoch_length)
    output = np.concatenate(
        [modified_spectrogram, np.tile(emg_log_rms, (EMG_COPIES, 1))]
    )
    return output


def get_mixture_values(
    img: np.array, labels: np.array, brain_state_set: BrainStateSet
) -> (np.array, np.array):
    """Compute weighted feature means and SDs for mixture z-scoring

    The outputs of this function can be used to standardize features
    extracted from all recordings from one subject under the same
    recording conditions. Note that labels must be in "class" format
    (i.e., integers between 0 and the number of scored states).

    :param img: combined EEG + EMG image - see create_eeg_emg_image()
    :param labels: brain state labels, in "class" format
    :param brain_state_set: set of brain state options
    :return: mixture means, mixture standard deviations
    """

    means = list()
    variances = list()
    mixture_weights = brain_state_set.mixture_weights

    # get feature means, variances by class
    for i in range(brain_state_set.n_classes):
        means.append(np.mean(img[:, labels == i], axis=1))
        variances.append(np.var(img[:, labels == i], axis=1))
    means = np.array(means)
    variances = np.array(variances)

    # mixture means are just weighted averages across classes
    mixture_means = means.T @ mixture_weights
    # mixture variance is given by the law of total variance
    mixture_sds = np.sqrt(
        variances.T @ mixture_weights
        + (
            (mixture_means - np.tile(mixture_means, (brain_state_set.n_classes, 1)))
            ** 2
        ).T
        @ mixture_weights
    )

    return mixture_means, mixture_sds


def mixture_z_score_img(
    img: np.array,
    brain_state_set: BrainStateSet,
    labels: np.array = None,
    mixture_means: np.array = None,
    mixture_sds: np.array = None,
) -> np.array:
    """Perform mixture z-scoring on a combined EEG+EMG image

    If brain state labels are provided, they will be used to calculate
    mixture means and SDs. Otherwise, you must provide those inputs.
    Note that pixel values in the output are in the 0-1 range and will
    clip z-scores beyond ABS_MAX_Z_SCORE.

    :param img: combined EEG + EMG image - see create_eeg_emg_image()
    :param brain_state_set: set of brain state options
    :param labels: labels, in "class" format
    :param mixture_means: mixture means
    :param mixture_sds: mixture standard deviations
    :return:
    """
    if labels is None and (mixture_means is None or mixture_sds is None):
        raise Exception("must provide either labels or mixture means+SDs")
    if labels is not None and ((mixture_means is not None) ^ (mixture_sds is not None)):
        warnings.warn("labels were given, mixture means / SDs will be ignored")

    if labels is not None:
        mixture_means, mixture_sds = get_mixture_values(
            img=img, labels=labels, brain_state_set=brain_state_set
        )

    img = ((img.T - mixture_means) / mixture_sds).T
    img = (img + ABS_MAX_Z_SCORE) / (2 * ABS_MAX_Z_SCORE)
    img = np.clip(img, 0, 1)

    return img


def format_img(img: np.array, epochs_per_img: int, add_padding: bool) -> np.array:
    """Adjust the format of an EEG+EMG image

    This function converts the values in a combined EEG+EMG image to uint8.
    This is a convenient format both for storing individual images as files,
    and for using the images as input to a classifier.
    This function also optionally adds new epochs to the beginning/end of the
    recording's image so that an image can be created for every epoch. For
    real-time scoring, padding should not be used.

    :param img: combined EEG + EMG image
    :param epochs_per_img: number of epochs in each individual image
    :param add_padding: whether to pad each side by (epochs_per_img - 1) / 2
    :return: formatted EEG + EMG image
    """
    # pad beginning and end
    if add_padding:
        pad_width = round((epochs_per_img - 1) / 2)
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
    epochs_per_img: int,
    brain_state_set: BrainStateSet,
    model_type: str,
) -> list[int]:
    """Create training dataset

    By default, the current epoch is located in the central column
    of pixels in each image. For real-time scoring applications,
    the current epoch is at the right edge of each image.

    :param recordings: list of recordings in the training set
    :param output_path: where to store training images
    :param epoch_length: epoch length, in seconds
    :param epochs_per_img: # number of epochs shown in each image
    :param brain_state_set: set of brain state options
    :param model_type: default or real-time
    :return: list of the names of any recordings that could not
            be used to create training images.
    """
    # recordings that had to be skipped
    failed_recordings = list()
    # image filenames for valid epochs
    filenames = list()
    # all valid labels from all valid recordings
    all_labels = list()
    # try to load each recording and create training images
    for i in trange(len(recordings)):
        recording = recordings[i]
        try:
            eeg, emg = load_recording(recording.recording_file)
            sampling_rate = recording.sampling_rate
            eeg, emg, sampling_rate = resample_and_standardize(
                eeg=eeg,
                emg=emg,
                sampling_rate=sampling_rate,
                epoch_length=epoch_length,
            )

            labels = load_labels(recording.label_file)
            labels = brain_state_set.convert_digit_to_class(labels)
            img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)
            img = mixture_z_score_img(
                img=img, brain_state_set=brain_state_set, labels=labels
            )
            img = format_img(img=img, epochs_per_img=epochs_per_img, add_padding=True)

            # the model type determines which epochs are used in each image
            if model_type == DEFAULT_MODEL_TYPE:
                # here, j is the index of the current epoch in 'labels'
                # and the index of the leftmost epoch in 'img'
                for j in range(img.shape[1] - (epochs_per_img - 1)):
                    if labels[j] is None:
                        continue
                    im = img[:, j : (j + epochs_per_img)]
                    filename = f"recording_{recording.name}_{j}_{labels[j]}.png"
                    filenames.append(filename)
                    all_labels.append(labels[j])
                    Image.fromarray(im).save(os.path.join(output_path, filename))
            else:
                # here, j is the index of the current epoch in 'labels'
                # but we throw away a few epochs at the start since they
                # would require even more padding on the left side.
                one_side_padding = round((epochs_per_img - 1) / 2)
                for j in range(one_side_padding, len(labels)):
                    if labels[j] is None:
                        continue
                    im = img[:, (j - one_side_padding) : j + one_side_padding + 1]
                    filename = f"recording_{recording.name}_{j}_{labels[j]}.png"
                    filenames.append(filename)
                    all_labels.append(labels[j])
                    Image.fromarray(im).save(os.path.join(output_path, filename))

        except Exception as e:
            print(e)
            failed_recordings.append(recording.name)

    # annotation file containing info on all images
    pd.DataFrame({FILENAME_COL: filenames, LABEL_COL: all_labels}).to_csv(
        os.path.join(output_path, ANNOTATIONS_FILENAME),
        index=False,
    )

    return failed_recordings
