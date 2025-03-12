import os

import pandas as pd
import scipy.io
import torch

import accusleepy.utils.constants as c
from accusleepy.utils.models import SSANN


def load_mat_files(file_path):
    eeg = scipy.io.loadmat(os.path.join(file_path, "EEG.mat"))["EEG"].T[0]
    emg = scipy.io.loadmat(os.path.join(file_path, "EMG.mat"))["EMG"].T[0]
    labels = scipy.io.loadmat(os.path.join(file_path, "labels.mat"))["labels"].T[0]
    return eeg, emg, labels


def load_calibration_file(filename):
    df = pd.read_csv(filename)
    mixture_means = df[c.MIXTURE_MEAN_COL].values
    mixture_sds = df[c.MIXTURE_SD_COL].values
    return mixture_means, mixture_sds


def save_model(model, output_dir, filename):
    torch.save(model.state_dict(), os.path.join(output_dir, filename) + ".pth")


def load_model(file_path):
    model = SSANN()
    model.load_state_dict(torch.load(file_path, weights_only=True))
    return model


def load_recording(file_path):
    df = pd.read_parquet(file_path)
    eeg = df[c.EEG_COL]
    emg = df[c.EMG_COL]
    return eeg, emg
