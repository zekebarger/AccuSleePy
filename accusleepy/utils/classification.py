import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io
from scipy.signal import ShortTimeFFT, windows, butter, filtfilt
import torch

from torch.utils.data import Dataset, DataLoader

# from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from accusleepy.utils.constants import (
    EMG_COPIES,
    N_CLASSES,
    FILENAME_COL,
    LABEL_COL,
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
        # TODO: labels start at 1...
        label = self.img_labels.at[idx, LABEL_COL] - 1
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
        self.fc1 = nn.Linear(int(32 * IMAGE_HEIGHT / 8), N_CLASSES)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch # needed?
        return self.fc1(x)


def train_model(annotations_file, img_dir):
    training_data = AccuSleepImageDataset(
        annotations_file=annotations_file,
        img_dir=img_dir,
    )
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    model = SSANN().to(device)

    criterion = nn.CrossEntropyLoss()
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

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
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
