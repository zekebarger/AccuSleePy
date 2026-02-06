"""Example real-time sleep scoring function."""

import numpy as np
import torch

from accusleepy.brain_state_set import BrainStateSet
from accusleepy.classification import get_device
from accusleepy.fileio import EMGFilter
from accusleepy.models import SSANN
from accusleepy.signal_processing import (
    create_eeg_emg_image,
    format_img,
    mixture_z_score_img,
)


def example_real_time_scoring_function(
    model: SSANN,
    eeg: np.ndarray,
    emg: np.ndarray,
    mixture_means: np.ndarray,
    mixture_sds: np.ndarray,
    sampling_rate: int | float,
    epoch_length: int | float,
    epochs_per_img: int,
    brain_state_set: BrainStateSet,
    emg_filter: EMGFilter,
) -> int:
    """Example function that could be used for real-time scoring

    This function demonstrates how you could use a model trained in
    "real-time" mode (current epoch on the right side of each image)
    to score incoming data. By passing a segment of EEG/EMG data
    into this function, the most recent epoch will be scored. For
    example, if the model expects 9 epochs worth of data and the
    epoch length is 5 seconds, you would pass in 45 seconds of data
    and would obtain the brain state of the most recent 5 seconds.

    Note:
    - The EEG and EMG signals must have length equal to
        sampling_rate * epoch_length * <number of epochs per image>.
    - The number of samples per epoch must be an integer.
    - This is just a demonstration, you should customize this for
        your application and there are probably ways to make it
        run faster.

    :param model: classification model
    :param eeg: EEG signal
    :param emg: EMG signal
    :param mixture_means: mixture means, for calibration
    :param mixture_sds: mixture standard deviations, for calibration
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :param epochs_per_img: number of epochs shown to the model at once
    :param brain_state_set: set of brain state options
    :param emg_filter: EMG filter parameters
    :return: brain state label
    """
    # prepare model
    # (this could be done outside the function)
    device = get_device()
    model = model.to(device)
    model.eval()

    # create and scale eeg+emg spectrogram
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length, emg_filter)
    img, _ = mixture_z_score_img(
        img,
        mixture_means=mixture_means,
        mixture_sds=mixture_sds,
        brain_state_set=brain_state_set,
    )
    img = format_img(img=img, epochs_per_img=epochs_per_img, add_padding=False)

    # create dataset for inference
    images = torch.from_numpy(np.array([img.astype("float32")]))
    images = images[:, None, :, :]  # add channel
    images = images.to(device)

    # perform classification
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    label = int(brain_state_set.convert_class_to_digit(predicted.cpu().numpy())[0])
    return label
