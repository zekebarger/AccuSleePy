import os

import pandas as pd
import scipy.io
import torch

from accusleepy.utils.constants import MIXTURE_MEAN_COL, MIXTURE_SD_COL
from accusleepy.utils.models import SSANN


def load_files(file_path):
    eeg = scipy.io.loadmat(os.path.join(file_path, "EEG.mat"))["EEG"].T[0]
    emg = scipy.io.loadmat(os.path.join(file_path, "EMG.mat"))["EMG"].T[0]
    labels = scipy.io.loadmat(os.path.join(file_path, "labels.mat"))["labels"].T[0]
    return eeg, emg, labels


def load_calibration_file(filename):
    df = pd.read_csv(filename)
    mixture_means = df[MIXTURE_MEAN_COL].values
    mixture_sds = df[MIXTURE_SD_COL].values
    return mixture_means, mixture_sds


def save_model(model, output_dir, filename):
    torch.save(model.state_dict(), os.path.join(output_dir, filename) + ".pth")


def load_model(file_path):
    model = SSANN()
    model.load_state_dict(torch.load(file_path, weights_only=True))
    return model
