import os

import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset, DataLoader

from torchvision.io import read_image
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from accusleepy.utils.constants import (
    EMG_COPIES,
    EPOCHS_PER_IMG,
    FILENAME_COL,
    LABEL_COL,
    MIXTURE_WEIGHTS,
    MIXTURE_MEAN_COL,
    MIXTURE_SD_COL,
    BRAIN_STATE_MAPPER,
)
from accusleepy.utils.signal_processing import (
    create_eeg_emg_image,
    get_mixture_values,
    mixture_z_score_img,
    truncate_signals,
    format_img,
)


BATCH_SIZE = 64
IMAGE_HEIGHT = 175 + EMG_COPIES  # TODO determine based on img size?

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
        img_path = os.path.join(self.img_dir, self.img_labels.at[idx, FILENAME_COL])
        image = read_image(img_path)
        label = self.img_labels.at[idx, LABEL_COL]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class SSANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(int(32 * IMAGE_HEIGHT / 8), BRAIN_STATE_MAPPER.n_classes)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch # needed?
        return self.fc1(x)


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

    weight = torch.tensor((MIXTURE_WEIGHTS**-1).astype("float32")).to(device)
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


def test_model(model, annotations_file, img_dir):
    test_data = AccuSleepImageDataset(
        annotations_file=annotations_file,
        img_dir=img_dir,
    )
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model = model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_dataloader:
            inputs, labels = data
            (inputs, labels) = (inputs.to(device), labels.to(device))
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"test accuracy: {correct / total:.2%}")


def save_model(model, output_dir, filename):
    torch.save(model.state_dict(), os.path.join(output_dir, filename) + ".pth")


def load_model(file_path):
    model = SSANN()
    model.load_state_dict(torch.load(file_path, weights_only=True))
    return model


def score_recording(
    model,
    eeg,
    emg,
    mixture_means,
    mixture_sds,
    sampling_rate,
    epoch_length,
    epochs_per_img=EPOCHS_PER_IMG,
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

    labels = BRAIN_STATE_MAPPER.convert_class_to_digit(predicted.cpu().numpy())
    return labels


def create_calibration_file(filename, eeg, emg, labels, sampling_rate, epoch_length):
    # labels = DIGITS

    eeg, emg = truncate_signals(eeg, emg, sampling_rate, epoch_length)
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length)
    mixture_means, mixture_sds = get_mixture_values(
        img, BRAIN_STATE_MAPPER.convert_digit_to_class(labels)
    )
    df = pd.DataFrame({MIXTURE_MEAN_COL: mixture_means, MIXTURE_SD_COL: mixture_sds})
    df.to_csv(filename, index=False)


def load_calibration_file(filename):
    df = pd.read_csv(filename)
    mixture_means = df[MIXTURE_MEAN_COL].values
    mixture_sds = df[MIXTURE_SD_COL].values
    return mixture_means, mixture_sds
