import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

import accusleepy.utils.constants as c

from accusleepy.utils.models import SSANN
from accusleepy.utils.signal_processing import (
    create_eeg_emg_image,
    format_img,
    get_mixture_values,
    mixture_z_score_img,
    truncate_signals,
)

BATCH_SIZE = 64

LEARNING_RATE = 1e-3
MOMENTUM = 0.9
EPOCHS = 6


class AccuSleepImageDataset(Dataset):
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
        img_path = os.path.join(self.img_dir, self.img_labels.at[idx, c.FILENAME_COL])
        image = read_image(img_path)
        label = self.img_labels.at[idx, c.LABEL_COL]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_device():
    return (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )


def train_model(annotations_file, img_dir):
    training_data = AccuSleepImageDataset(
        annotations_file=annotations_file,
        img_dir=img_dir,
    )
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    device = get_device()
    model = SSANN().to(device)

    weight = torch.tensor((c.MIXTURE_WEIGHTS**-1).astype("float32")).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    model.train()

    for epoch in range(EPOCHS):
        for data in train_dataloader:
            inputs, labels = data
            (inputs, labels) = (inputs.to(device), labels.to(device))

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def test_model(
    model,
    eeg,
    emg,
    labels,
    sampling_rate,
    epoch_length,
    epochs_per_img=c.EPOCHS_PER_IMG,
):
    eeg, emg = truncate_signals(eeg, emg, sampling_rate, epoch_length)
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)
    mixture_means, mixture_sds = get_mixture_values(
        img, c.BRAIN_STATE_MAPPER.convert_digit_to_class(labels)
    )
    pred = score_recording(
        model,
        eeg,
        emg,
        mixture_means,
        mixture_sds,
        sampling_rate,
        epoch_length,
        epochs_per_img=epochs_per_img,
    )

    print(f"test accuracy: {sum(pred == labels) / len(labels):.2%}")


def score_recording(
    model,
    eeg,
    emg,
    mixture_means,
    mixture_sds,
    sampling_rate,
    epoch_length,
    epochs_per_img=c.EPOCHS_PER_IMG,
):
    # prepare model
    device = get_device()
    model = model.to(device)
    model.eval()

    # preprocess eeg, emg
    eeg, emg = truncate_signals(eeg, emg, sampling_rate, epoch_length)

    # create and scale eeg+emg spectrogram
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)
    img = mixture_z_score_img(img, mixture_means=mixture_means, mixture_sds=mixture_sds)
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

    labels = c.BRAIN_STATE_MAPPER.convert_class_to_digit(predicted.cpu().numpy())
    return labels


def create_calibration_file(filename, eeg, emg, labels, sampling_rate, epoch_length):
    # labels = DIGITS

    eeg, emg = truncate_signals(eeg, emg, sampling_rate, epoch_length)
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)
    mixture_means, mixture_sds = get_mixture_values(
        img, c.BRAIN_STATE_MAPPER.convert_digit_to_class(labels)
    )
    df = pd.DataFrame(
        {c.MIXTURE_MEAN_COL: mixture_means, c.MIXTURE_SD_COL: mixture_sds}
    )
    df.to_csv(filename, index=False)
