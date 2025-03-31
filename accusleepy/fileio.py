import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.io
import torch
from PySide6.QtWidgets import QListWidgetItem

from accusleepy.brain_state_set import BRAIN_STATES_KEY, BrainState, BrainStateSet
from accusleepy.constants import (
    BRAIN_STATE_COL,
    CONFIG_FILE,
    EEG_COL,
    EMG_COL,
    MIXTURE_MEAN_COL,
    MIXTURE_SD_COL,
    RECORDING_LIST_NAME,
    UNDEFINED_LABEL,
)
from accusleepy.models import SSANN


@dataclass
class Recording:
    """Store information about a recording"""

    name: int = 1  # name to show in the GUI
    recording_file: str = ""  # path to recording file
    label_file: str = ""  # path to label file
    calibration_file: str = ""  # path to calibration file
    sampling_rate: int | float = 0.0  # sampling rate, in Hz
    widget: QListWidgetItem = None  # list item widget shown in the GUI


def load_calibration_file(filename: str) -> (np.array, np.array):
    """Load a calibration file

    :param filename: filename
    :return: mixture means and SDs
    """
    df = pd.read_csv(filename)
    mixture_means = df[MIXTURE_MEAN_COL].values
    mixture_sds = df[MIXTURE_SD_COL].values
    return mixture_means, mixture_sds


def save_model(
    model: SSANN,
    filename: str,
    epoch_length: int | float,
    epochs_per_img: int,
    model_type: str,
    brain_state_set: BrainStateSet,
) -> None:
    """Save classification model and its metadata

    :param model: classification model
    :param epoch_length: epoch length used when training the model
    :param epochs_per_img: number of epochs in each model input
    :param model_type: default or real-time
    :param brain_state_set: set of brain state options
    :param filename: filename
    """
    state_dict = model.state_dict()
    state_dict.update({"epoch_length": epoch_length})
    state_dict.update({"epochs_per_img": epochs_per_img})
    state_dict.update({"model_type": model_type})
    state_dict.update(
        {BRAIN_STATES_KEY: brain_state_set.to_output_dict()[BRAIN_STATES_KEY]}
    )

    torch.save(state_dict, filename)


def load_model(filename: str) -> tuple[SSANN, int | float, int, str, dict]:
    """Load classification model and its metadata

    :param filename: filename
    :return: model, epoch length used when training the model,
        number of epochs in each model input, model type
        (default or real-time), set of brain state options
        used when training the model
    """
    state_dict = torch.load(filename, weights_only=True)
    epoch_length = state_dict.pop("epoch_length")
    epochs_per_img = state_dict.pop("epochs_per_img")
    model_type = state_dict.pop("model_type")
    brain_states = state_dict.pop(BRAIN_STATES_KEY)
    n_classes = len([b for b in brain_states if b["is_scored"]])

    model = SSANN(n_classes=n_classes)
    model.load_state_dict(state_dict)
    return model, epoch_length, epochs_per_img, model_type, brain_states


def load_csv_or_parquet(filename: str) -> pd.DataFrame:
    """Load a csv or parquet file as a dataframe

    :param filename: filename
    :return: dataframe of file contents
    """
    extension = os.path.splitext(filename)[1]
    if extension == ".csv":
        df = pd.read_csv(filename)
    elif extension == ".parquet":
        df = pd.read_parquet(filename)
    else:
        raise Exception("file must be csv or parquet")
    return df


def load_recording(filename: str) -> (np.array, np.array):
    """Load recording of EEG and EMG time series data

    :param filename: filename
    :return: arrays of EEG and EMG data
    """
    df = load_csv_or_parquet(filename)
    eeg = df[EEG_COL].values
    emg = df[EMG_COL].values
    return eeg, emg


def load_labels(filename: str) -> np.array:
    """Load file of brain state labels

    :param filename: filename
    :return: array of brain state labels
    """
    df = load_csv_or_parquet(filename)
    return df[BRAIN_STATE_COL].values


def save_labels(labels: np.array, filename: str) -> None:
    """Save brain state labels to file

    :param labels: brain state labels
    :param filename: filename
    """
    pd.DataFrame({BRAIN_STATE_COL: labels}).to_csv(filename, index=False)


def load_config() -> BrainStateSet:
    """Load configuration file with brain state options

    :return: set of brain state options
    """
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE), "r"
    ) as f:
        data = json.load(f)
    return BrainStateSet(
        [BrainState(**b) for b in data[BRAIN_STATES_KEY]], UNDEFINED_LABEL
    )


def save_config(brain_state_set: BrainStateSet) -> None:
    """Save configuration of brain state options to json file

    :param brain_state_set: set of brain state options
    """
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE), "w"
    ) as f:
        json.dump(brain_state_set.to_output_dict(), f, indent=4)


def load_recording_list(filename: str) -> list[Recording]:
    """Load list of recordings from file

    :param filename: filename of list of recordings
    :return: list of recordings
    """
    with open(filename, "r") as f:
        data = json.load(f)
    recording_list = [Recording(**r) for r in data[RECORDING_LIST_NAME]]
    for i, r in enumerate(recording_list):
        r.name = i + 1
    return recording_list


def save_recording_list(filename: str, recordings: list[Recording]) -> None:
    """Save list of recordings to file

    :param filename: where to save the list
    :param recordings: list of recordings to export
    """
    recording_dict = {
        RECORDING_LIST_NAME: [
            {
                "recording_file": r.recording_file,
                "label_file": r.label_file,
                "calibration_file": r.calibration_file,
                "sampling_rate": r.sampling_rate,
            }
            for r in recordings
        ]
    }
    with open(filename, "w") as f:
        json.dump(recording_dict, f, indent=4)


def convert_mat_files(path: str) -> None:
    eeg, emg, labels = load_mat_files(path)
    pd.DataFrame({EEG_COL: eeg, EMG_COL: emg}).to_parquet(
        os.path.join(path, "recording.parquet")
    )
    pd.DataFrame({BRAIN_STATE_COL: labels}).to_csv(
        os.path.join(path, "labels.csv"), index=False
    )


def load_mat_files(file_path: str) -> (np.array, np.array, np.array):
    eeg = scipy.io.loadmat(os.path.join(file_path, "EEG.mat"))["EEG"].T[0]
    emg = scipy.io.loadmat(os.path.join(file_path, "EMG.mat"))["EMG"].T[0]
    labels = scipy.io.loadmat(os.path.join(file_path, "labels.mat"))["labels"].T[0]
    return eeg, emg, labels
