import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

import accusleepy.constants as c
from accusleepy.brain_state_set import BrainStateSet
from accusleepy.fileio import Recording, load_config, load_labels, load_recording
from accusleepy.models import SSANN
from accusleepy.signal_processing import (
    create_eeg_emg_image,
    format_img,
    get_mixture_values,
    mixture_z_score_img,
    resample_and_standardize,
)

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
TRAINING_EPOCHS = 6


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


def train_model(
    annotations_file: str,
    img_dir: str,
    epochs_per_image: int,
    model_type: str,
    mixture_weights: np.array,
    n_classes: int,
) -> SSANN:
    """Train a classification model for sleep scoring

    :param annotations_file: file with information on each training image
    :param img_dir: training image location
    :param epochs_per_image: number of epochs per image
    :param model_type: default or real-time
    :param mixture_weights: typical relative frequencies of brain states
    :param n_classes: number of classes the model will learn
    :return: trained Sleep Scoring Artificial Neural Network model
    """
    training_data = AccuSleepImageDataset(
        annotations_file=annotations_file,
        img_dir=img_dir,
    )
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    device = get_device()
    model = SSANN(n_classes=n_classes)
    # store useful info in the weights
    model.epochs_per_image = torch.nn.Parameter(
        torch.Tensor([epochs_per_image]), requires_grad=False
    )
    model.model_type = torch.nn.Parameter(
        torch.Tensor([c.MODEL_TYPE_TO_KEY[model_type]]), requires_grad=False
    )
    model.to(device)
    model.train()

    # correct for class imbalance
    weight = torch.tensor((mixture_weights**-1).astype("float32")).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    for epoch in range(TRAINING_EPOCHS):
        print(f"Starting training epoch {epoch + 1}/{TRAINING_EPOCHS}")
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
    eeg: np.array,
    emg: np.array,
    mixture_means: np.array,
    mixture_sds: np.array,
    sampling_rate: int | float,
    epoch_length: int | float,
    brain_state_set: BrainStateSet,
) -> np.array:
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
    :param brain_state_set: set of brain state options
    :return: brain state labels
    """
    # prepare model
    device = get_device()
    model = model.to(device)
    model.eval()
    epochs_per_img = int(model.epochs_per_image.item())

    # create and scale eeg+emg spectrogram
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)
    img = mixture_z_score_img(
        img,
        mixture_means=mixture_means,
        mixture_sds=mixture_sds,
        brain_state_set=brain_state_set,
    )
    img = format_img(img, epochs_per_img)

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
        _, predicted = torch.max(outputs, 1)

    labels = brain_state_set.convert_class_to_digit(predicted.cpu().numpy())
    return labels


def create_calibration_file(
    filename: str,
    eeg: np.array,
    emg: np.array,
    labels: np.array,
    sampling_rate: int | float,
    epoch_length: int | float,
    brain_state_set: BrainStateSet,
) -> None:
    """Create file of calibration data for a subject

    This assumes signals have been preprocessed to contain an integer
    number of epochs.

    :param filename: filename for the calibration file
    :param eeg: EEG signal
    :param emg: EMG signal
    :param labels: brain state labels, as digits
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :param brain_state_set: set of brain state options
    """
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)
    mixture_means, mixture_sds = get_mixture_values(
        img=img,
        labels=brain_state_set.convert_digit_to_class(labels),
        brain_state_set=brain_state_set,
    )
    pd.DataFrame(
        {c.MIXTURE_MEAN_COL: mixture_means, c.MIXTURE_SD_COL: mixture_sds}
    ).to_csv(filename, index=False)


def test_model(
    model: SSANN,
    recordings: list[Recording],
    epoch_length: int | float,
) -> None:
    all_labels = np.empty(0).astype(int)
    all_predictions = np.empty(0).astype(int)

    brain_state_set = load_config()

    for recording in recordings:
        eeg, emg = load_recording(recording.recording_file)
        labels = load_labels(recording.label_file)

        eeg, emg, recording.sampling_rate = resample_and_standardize(
            eeg=eeg,
            emg=emg,
            sampling_rate=recording.sampling_rate,
            epoch_length=epoch_length,
        )
        img = create_eeg_emg_image(eeg, emg, recording.sampling_rate, epoch_length)
        mixture_means, mixture_sds = get_mixture_values(
            img=img,
            labels=brain_state_set.convert_digit_to_class(labels),
            brain_state_set=brain_state_set,
        )
        pred = score_recording(
            model,
            eeg,
            emg,
            mixture_means,
            mixture_sds,
            recording.sampling_rate,
            epoch_length,
            brain_state_set,
        )

        all_labels = np.concatenate([all_labels, labels])
        all_predictions = np.concatenate([all_predictions, pred])

    print(
        f"test accuracy: {np.sum(all_predictions == all_labels) / len(all_labels):.2%}"
    )
