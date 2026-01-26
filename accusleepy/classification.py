import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm import trange

import accusleepy.constants as c
from accusleepy.brain_state_set import BrainStateSet
from accusleepy.fileio import EMGFilter, Hyperparameters
from accusleepy.models import SSANN
from accusleepy.signal_processing import (
    create_eeg_emg_image,
    format_img,
    get_mixture_values,
    mixture_z_score_img,
)


class AccuSleepImageDataset(Dataset):
    """Dataset for loading AccuSleep training images"""

    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = str(
            os.path.join(self.img_dir, self.img_labels.at[idx, c.FILENAME_COL])
        )
        image = read_image(img_path)
        label = self.img_labels.at[idx, c.LABEL_COL]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_device():
    """Get accelerator, if one is available"""
    return (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )


def create_dataloader(
    annotations_file: str,
    img_dir: str,
    hyperparameters: Hyperparameters,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for a dataset of training or calibration images

    :param annotations_file: file with information on each training image
    :param img_dir: training image location
    :param hyperparameters: model training hyperparameters
    :param shuffle: reshuffle data for every epoch
    :return: DataLoader for the data

    """
    image_dataset = AccuSleepImageDataset(
        annotations_file=annotations_file,
        img_dir=img_dir,
    )
    return DataLoader(
        image_dataset, batch_size=hyperparameters.batch_size, shuffle=shuffle
    )


def train_ssann(
    annotations_file: str,
    img_dir: str,
    training_class_balance: np.ndarray,
    n_classes: int,
    hyperparameters: Hyperparameters,
) -> SSANN:
    """Train a SSANN classification model for sleep scoring

    :param annotations_file: file with information on each training image
    :param img_dir: training image location
    :param training_class_balance: proportion of each class in the training set
    :param n_classes: number of classes the model will learn
    :param hyperparameters: model training hyperparameters
    :return: trained Sleep Scoring Artificial Neural Network model
    """
    train_dataloader = create_dataloader(
        annotations_file=annotations_file,
        img_dir=img_dir,
        hyperparameters=hyperparameters,
    )

    device = get_device()
    model = SSANN(n_classes=n_classes)
    model.to(device)
    model.train()

    # correct for class imbalance
    weight = torch.tensor((training_class_balance**-1).astype("float32")).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        momentum=hyperparameters.momentum,
    )

    for _ in trange(hyperparameters.training_epochs):
        for data in train_dataloader:
            inputs, labels = data
            (inputs, labels) = (inputs.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def score_recording(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Use classification model to get brain state labels for a recording

    This assumes signals have been preprocessed to contain an integer
    number of epochs.

    :param model: classification model
    :param eeg: EEG signal
    :param emg: EMG signal
    :param mixture_means: mixture means, for calibration
    :param mixture_sds: mixture standard deviations, for calibration
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :param epochs_per_img: number of epochs for the model to consider
    :param brain_state_set: set of brain state options
    :param emg_filter: EMG filter parameters
    :return: brain state labels, confidence scores
    """
    # prepare model
    device = get_device()
    model = model.to(device)
    model.eval()

    # create and scale eeg+emg spectrogram
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length, emg_filter)
    img = mixture_z_score_img(
        img,
        mixture_means=mixture_means,
        mixture_sds=mixture_sds,
        brain_state_set=brain_state_set,
    )
    img = format_img(img=img, epochs_per_img=epochs_per_img, add_padding=True)

    # create dataset for inference
    images = []
    for i in range(img.shape[1] - epochs_per_img + 1):
        images.append(img[:, i : (i + epochs_per_img)].astype("float32"))
    images = torch.from_numpy(np.array(images))
    images = images[:, None, :, :]  # add channel
    images = images.to(device)

    # perform classification
    with torch.no_grad():
        outputs = model(images)
        logits, predicted = torch.max(outputs, 1)

    labels = brain_state_set.convert_class_to_digit(predicted.cpu().numpy())
    confidence_scores = 1 / (1 + np.e ** (-logits.cpu().numpy()))

    return labels, confidence_scores


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
    # this could be done outside the function
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


def create_calibration_file(
    filename: str,
    eeg: np.ndarray,
    emg: np.ndarray,
    labels: np.ndarray,
    sampling_rate: int | float,
    epoch_length: int | float,
    brain_state_set: BrainStateSet,
    emg_filter: EMGFilter,
) -> bool:
    """Create file of calibration data for a subject

    Returns True if any features derived from the recording
    have 0 variance.

    This assumes that EEG and EMG signals have been preprocessed to
    contain an integer number of epochs and that there are a
    sufficient number of labeled epochs for each scored brain state.

    :param filename: filename for the calibration file
    :param eeg: EEG signal
    :param emg: EMG signal
    :param labels: brain state labels, as digits
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :param brain_state_set: set of brain state options
    :param emg_filter: EMG filter parameters
    :return: whether zero-variance features were detected
    """
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length, emg_filter)
    mixture_means, mixture_sds = get_mixture_values(
        img=img,
        labels=brain_state_set.convert_digit_to_class(labels),
        brain_state_set=brain_state_set,
    )
    had_zero_variance = np.any(mixture_sds == 0)
    pd.DataFrame(
        {c.MIXTURE_MEAN_COL: mixture_means, c.MIXTURE_SD_COL: mixture_sds}
    ).to_csv(filename, index=False)
    return had_zero_variance
