import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QListWidgetItem

from accusleepy.brain_state_set import BRAIN_STATES_KEY, BrainState, BrainStateSet
import accusleepy.constants as c


@dataclass
class EMGFilter:
    """Convenience class for a EMG filter parameters"""

    order: int  # filter order
    bp_lower: int | float  # lower bandpass frequency
    bp_upper: int | float  # upper bandpass frequency


@dataclass
class Hyperparameters:
    """Convenience class for model training hyperparameters"""

    batch_size: int
    learning_rate: float
    momentum: float
    training_epochs: int


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
    mixture_means = df[c.MIXTURE_MEAN_COL].values
    mixture_sds = df[c.MIXTURE_SD_COL].values
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
    eeg = df[c.EEG_COL].values
    emg = df[c.EMG_COL].values
    return eeg, emg


def load_labels(filename: str) -> (np.array, np.array):
    """Load file of brain state labels and confidence scores

    :param filename: filename
    :return: array of brain state labels and, optionally, array of confidence scores
    """
    df = load_csv_or_parquet(filename)
    if c.CONFIDENCE_SCORE_COL in df.columns:
        return df[c.BRAIN_STATE_COL].values, df[c.CONFIDENCE_SCORE_COL].values
    else:
        return df[c.BRAIN_STATE_COL].values, None


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
            {c.BRAIN_STATE_COL: labels, c.CONFIDENCE_SCORE_COL: confidence_scores}
        ).to_csv(filename, index=False)
    else:
        pd.DataFrame({c.BRAIN_STATE_COL: labels}).to_csv(filename, index=False)


def load_config() -> tuple[
    BrainStateSet, int | float, bool, bool, int | float, EMGFilter, Hyperparameters
]:
    """Load configuration file with brain state options

    :return: set of brain state options,
        default epoch length,
        default overwrite setting,
        default confidence score output setting,
        default minimum bout length,
        EMG filter parameters,
        model training hyperparameters
    """
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), c.CONFIG_FILE), "r"
    ) as f:
        data = json.load(f)

    return (
        BrainStateSet(
            [BrainState(**b) for b in data[BRAIN_STATES_KEY]], c.UNDEFINED_LABEL
        ),
        data[c.DEFAULT_EPOCH_LENGTH_KEY],
        data.get(c.DEFAULT_OVERWRITE_KEY, c.DEFAULT_OVERWRITE_SETTING),
        data.get(c.DEFAULT_CONFIDENCE_SETTING_KEY, c.DEFAULT_CONFIDENCE_SETTING),
        data.get(c.DEFAULT_MIN_BOUT_LENGTH_KEY, c.DEFAULT_MIN_BOUT_LENGTH),
        EMGFilter(
            **data.get(
                c.EMG_FILTER_KEY,
                {
                    "order": c.DEFAULT_EMG_FILTER_ORDER,
                    "bp_lower": c.DEFAULT_EMG_BP_LOWER,
                    "bp_upper": c.DEFAULT_EMG_BP_UPPER,
                },
            )
        ),
        Hyperparameters(
            **data.get(
                c.HYPERPARAMETERS_KEY,
                {
                    "batch_size": c.DEFAULT_BATCH_SIZE,
                    "learning_rate": c.DEFAULT_LEARNING_RATE,
                    "momentum": c.DEFAULT_MOMENTUM,
                    "training_epochs": c.DEFAULT_TRAINING_EPOCHS,
                },
            )
        ),
    )


def save_config(
    brain_state_set: BrainStateSet,
    default_epoch_length: int | float,
    overwrite_setting: bool,
    save_confidence_setting: bool,
    min_bout_length: int | float,
    emg_filter: EMGFilter,
    hyperparameters: Hyperparameters,
) -> None:
    """Save configuration of brain state options to json file

    :param brain_state_set: set of brain state options
    :param default_epoch_length: default epoch length
    :param save_confidence_setting: default setting for
        saving confidence scores
    :param emg_filter: EMG filter parameters
    :param min_bout_length: default minimum bout length
    :param overwrite_setting: default setting for overwriting
        existing labels
    :param hyperparameters: model training hyperparameters
    """
    output_dict = brain_state_set.to_output_dict()
    output_dict.update({c.DEFAULT_EPOCH_LENGTH_KEY: default_epoch_length})
    output_dict.update({c.DEFAULT_OVERWRITE_KEY: overwrite_setting})
    output_dict.update({c.DEFAULT_CONFIDENCE_SETTING_KEY: save_confidence_setting})
    output_dict.update({c.DEFAULT_MIN_BOUT_LENGTH_KEY: min_bout_length})
    output_dict.update({c.EMG_FILTER_KEY: emg_filter.__dict__})
    output_dict.update({c.HYPERPARAMETERS_KEY: hyperparameters.__dict__})
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), c.CONFIG_FILE), "w"
    ) as f:
        json.dump(output_dict, f, indent=4)


def load_recording_list(filename: str) -> list[Recording]:
    """Load list of recordings from file

    :param filename: filename of list of recordings
    :return: list of recordings
    """
    with open(filename, "r") as f:
        data = json.load(f)
    recording_list = [Recording(**r) for r in data[c.RECORDING_LIST_NAME]]
    for i, r in enumerate(recording_list):
        r.name = i + 1
    return recording_list


def save_recording_list(filename: str, recordings: list[Recording]) -> None:
    """Save list of recordings to file

    :param filename: where to save the list
    :param recordings: list of recordings to export
    """
    recording_dict = {
        c.RECORDING_LIST_NAME: [
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
