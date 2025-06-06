import numpy as np
from torch import device, flatten, nn
from torch import load as torch_load
from torch import save as torch_save

from accusleepy.brain_state_set import BRAIN_STATES_KEY, BrainStateSet
from accusleepy.constants import (
    DOWNSAMPLING_START_FREQ,
    EMG_COPIES,
    MIN_WINDOW_LEN,
    UPPER_FREQ,
)
from accusleepy.temperature_scaling import ModelWithTemperature

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
        x = self.pool(nn.functional.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.conv3_bn(self.conv3(x))))
        x = flatten(x, 1)  # flatten all dimensions except batch
        return self.fc1(x)


def save_model(
    model: SSANN,
    filename: str,
    epoch_length: int | float,
    epochs_per_img: int,
    model_type: str,
    brain_state_set: BrainStateSet,
    is_calibrated: bool,
) -> None:
    """Save classification model and its metadata

    :param model: classification model
    :param filename: filename
    :param epoch_length: epoch length used when training the model
    :param epochs_per_img: number of epochs in each model input
    :param model_type: default or real-time
    :param brain_state_set: set of brain state options
    :param is_calibrated: whether the model has been calibrated
    """
    state_dict = model.state_dict()
    state_dict.update({"epoch_length": epoch_length})
    state_dict.update({"epochs_per_img": epochs_per_img})
    state_dict.update({"model_type": model_type})
    state_dict.update({"is_calibrated": is_calibrated})
    state_dict.update(
        {BRAIN_STATES_KEY: brain_state_set.to_output_dict()[BRAIN_STATES_KEY]}
    )

    torch_save(state_dict, filename)


def load_model(filename: str) -> tuple[SSANN, int | float, int, str, dict]:
    """Load classification model and its metadata

    :param filename: filename
    :return: model, epoch length used when training the model,
        number of epochs in each model input, model type
        (default or real-time), set of brain state options
        used when training the model
    """
    state_dict = torch_load(filename, weights_only=True, map_location=device("cpu"))
    epoch_length = state_dict.pop("epoch_length")
    epochs_per_img = state_dict.pop("epochs_per_img")
    model_type = state_dict.pop("model_type")
    if "is_calibrated" in state_dict:
        is_calibrated = state_dict.pop("is_calibrated")
    else:
        is_calibrated = False
    brain_states = state_dict.pop(BRAIN_STATES_KEY)
    n_classes = len([b for b in brain_states if b["is_scored"]])

    model = SSANN(n_classes=n_classes)
    if is_calibrated:
        model = ModelWithTemperature(model)
    model.load_state_dict(state_dict)
    return model, epoch_length, epochs_per_img, model_type, brain_states
