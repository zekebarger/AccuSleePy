import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import accusleepy.utils.constants as c

IMAGE_HEIGHT = (
    len(np.arange(0, c.DOWNSAMPLING_START_FREQ, 1 / c.MIN_WINDOW_LEN))
    + len(np.arange(c.DOWNSAMPLING_START_FREQ, c.UPPER_FREQ, 2 / c.MIN_WINDOW_LEN))
    + c.EMG_COPIES
)


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
        self.fc1 = nn.Linear(int(32 * IMAGE_HEIGHT / 8), c.BRAIN_STATE_MAPPER.n_classes)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch # needed?
        return self.fc1(x)
