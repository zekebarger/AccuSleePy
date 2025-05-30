import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QListWidgetItem

from accusleepy.brain_state_set import BRAIN_STATES_KEY, BrainState, BrainStateSet
from accusleepy.constants import (
    BRAIN_STATE_COL,
    CONFIDENCE_SCORE_COL,
    CONFIG_FILE,
    DEFAULT_EPOCH_LENGTH_KEY,
    EEG_COL,
    EMG_COL,
    MIXTURE_MEAN_COL,
    MIXTURE_SD_COL,
    RECORDING_LIST_NAME,
    UNDEFINED_LABEL,
)


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


def load_labels(filename: str) -> Tuple[np.array, None] | Tuple[np.array, np.array]:
    """Load file of brain state labels

    :param filename: filename
    :return: array of brain state labels and, optionally, array of confidence scores
    """
    df = load_csv_or_parquet(filename)
    if CONFIDENCE_SCORE_COL in df.columns:
        return df[BRAIN_STATE_COL].values, df[CONFIDENCE_SCORE_COL].values
    else:
        return df[BRAIN_STATE_COL].values, None


def save_labels(
    labels: np.array, filename: str, confidence_scores: np.array = None
) -> None:
    """Save brain state labels to file

    :param labels: brain state labels
    :param filename: filename
    :param confidence_scores: optional confidence scores
    """
    if confidence_scores is not None:
        pd.DataFrame(
            {BRAIN_STATE_COL: labels, CONFIDENCE_SCORE_COL: confidence_scores}
        ).to_csv(filename, index=False)
    else:
        pd.DataFrame({BRAIN_STATE_COL: labels}).to_csv(filename, index=False)


def load_config() -> tuple[BrainStateSet, int | float]:
    """Load configuration file with brain state options

    :return: set of brain state options and default epoch length
    """
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE), "r"
    ) as f:
        data = json.load(f)
    return BrainStateSet(
        [BrainState(**b) for b in data[BRAIN_STATES_KEY]], UNDEFINED_LABEL
    ), data[DEFAULT_EPOCH_LENGTH_KEY]


def save_config(
    brain_state_set: BrainStateSet, default_epoch_length: int | float
) -> None:
    """Save configuration of brain state options to json file

    :param brain_state_set: set of brain state options
    :param default_epoch_length: epoch length to use when the GUI starts
    """
    output_dict = brain_state_set.to_output_dict()
    output_dict.update({DEFAULT_EPOCH_LENGTH_KEY: default_epoch_length})
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE), "w"
    ) as f:
        json.dump(output_dict, f, indent=4)


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
