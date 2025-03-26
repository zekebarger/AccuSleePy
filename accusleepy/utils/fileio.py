import os

import numpy as np
import pandas as pd
import scipy.io
import torch

import accusleepy.utils.constants as c
from accusleepy.utils.models import SSANN


def load_mat_files(file_path: str) -> (np.array, np.array, np.array):
    eeg = scipy.io.loadmat(os.path.join(file_path, "EEG.mat"))["EEG"].T[0]
    emg = scipy.io.loadmat(os.path.join(file_path, "EMG.mat"))["EMG"].T[0]
    labels = scipy.io.loadmat(os.path.join(file_path, "labels.mat"))["labels"].T[0]
    return eeg, emg, labels


def convert_mat_files(path: str) -> None:
    eeg, emg, labels = load_mat_files(path)
    pd.DataFrame({c.EEG_COL: eeg, c.EMG_COL: emg}).to_parquet(
        os.path.join(path, "recording.parquet")
    )
    pd.DataFrame({c.BRAIN_STATE_COL: labels}).to_csv(
        os.path.join(path, "labels.csv"), index=False
    )


def load_calibration_file(filename: str) -> (np.array, np.array):
    df = pd.read_csv(filename)
    mixture_means = df[c.MIXTURE_MEAN_COL].values
    mixture_sds = df[c.MIXTURE_SD_COL].values
    return mixture_means, mixture_sds


def save_model(model: SSANN, filename: str) -> None:
    torch.save(model.state_dict(), filename)


def load_model(file_path: str) -> SSANN:
    model = SSANN()
    model.load_state_dict(torch.load(file_path, weights_only=True))
    return model


# note: requires fastparquet
def load_csv_or_parquet(file_path: str) -> pd.DataFrame:
    extension = os.path.splitext(file_path)[1]
    if extension == ".csv":
        df = pd.read_csv(file_path)
    elif extension == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise Exception("file must be csv or parquet")
    return df


def load_recording(file_path: str) -> (np.array, np.array):
    df = load_csv_or_parquet(file_path)
    eeg = df[c.EEG_COL].values
    emg = df[c.EMG_COL].values
    return eeg, emg


def load_labels(file_path: str) -> np.array:
    df = load_csv_or_parquet(file_path)
    return df[c.BRAIN_STATE_COL].values


def save_labels(labels: np.array, file_path: str) -> None:
    pd.DataFrame({c.BRAIN_STATE_COL: labels}).to_csv(file_path, index=False)
