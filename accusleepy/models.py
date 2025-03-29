import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from accusleepy.constants import (
    DOWNSAMPLING_START_FREQ,
    EMG_COPIES,
    MIN_WINDOW_LEN,
    UPPER_FREQ,
)

# height in pixels of each training image
IMAGE_HEIGHT = (
    len(np.arange(0, DOWNSAMPLING_START_FREQ, 1 / MIN_WINDOW_LEN))
    + len(np.arange(DOWNSAMPLING_START_FREQ, UPPER_FREQ, 2 / MIN_WINDOW_LEN))
    + EMG_COPIES
)


class SSANN(nn.Module):
    """Small CNN for classifying images"""

    def __init__(self, n_classes: int):
        super().__init__()

        # useful custom parameters
        self.epochs_per_image = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.model_type = nn.Parameter(torch.Tensor(1), requires_grad=False)

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
        self.fc1 = nn.Linear(int(32 * IMAGE_HEIGHT / 8), n_classes)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return self.fc1(x)
