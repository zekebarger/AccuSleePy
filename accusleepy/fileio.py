import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.io
import torch
from PySide6.QtWidgets import QListWidgetItem

import accusleepy.config as c
from accusleepy.misc import BRAIN_STATES_KEY, BrainState, BrainStateMapper
from accusleepy.models import SSANN


@dataclass
class Recording:
    name: int = 1  # name to show in the GUI
    recording_file: str = ""  # path to recording file
    label_file: str = ""  # path to label file
    calibration_file: str = ""  # path to calibration file
    sampling_rate: int | float = 0.0  # sampling rate, in Hz
    widget: QListWidgetItem = None  # reference to widget shown in the GUI


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


def load_model(filename: str, n_classes: int) -> SSANN:
    model = SSANN(n_classes=n_classes)
    model.load_state_dict(torch.load(filename, weights_only=True))
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


def load_config() -> BrainStateMapper:
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), c.CONFIG_FILE), "r"
    ) as f:
        data = json.load(f)
    return BrainStateMapper(
        [BrainState(**b) for b in data[BRAIN_STATES_KEY]], c.UNDEFINED_LABEL
    )


def save_config(brain_state_mapper: BrainStateMapper) -> None:
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), c.CONFIG_FILE), "w"
    ) as f:
        json.dump(brain_state_mapper.output_dict(), f, indent=4)
