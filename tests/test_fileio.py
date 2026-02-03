"""Tests for accusleepy/fileio.py"""

import json
import os

import numpy as np
import pandas as pd
import pytest

import accusleepy.constants as c
from accusleepy.fileio import (
    AccuSleePyConfig,
    Recording,
    get_version,
    load_calibration_file,
    load_config,
    load_csv_or_parquet,
    load_labels,
    load_recording,
    load_recording_list,
    save_labels,
    save_recording_list,
)


def test_load_config(tmp_path, monkeypatch):
    """Test the configuration file loads successfully, copying default to user dir"""
    user_config = str(tmp_path / "config.json")
    monkeypatch.setattr("accusleepy.fileio._get_user_config_path", lambda: user_config)

    # User config doesn't exist yet — load_config should copy the default
    assert not os.path.exists(user_config)
    config = load_config()
    assert os.path.exists(user_config)
    assert isinstance(config, AccuSleePyConfig)
    assert type(config.epochs_to_show) is int


# Tests for load_csv_or_parquet


def test_load_csv_or_parquet_csv(tmp_path):
    """Load CSV file correctly"""
    csv_filename = str(tmp_path / "test.csv")
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    df.to_csv(csv_filename, index=False)

    result = load_csv_or_parquet(csv_filename)
    pd.testing.assert_frame_equal(result, df)


def test_load_csv_or_parquet_parquet(tmp_path):
    """Load Parquet file correctly"""
    parquet_filename = str(tmp_path / "test.parquet")
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    df.to_parquet(parquet_filename, index=False)

    result = load_csv_or_parquet(parquet_filename)
    pd.testing.assert_frame_equal(result, df)


def test_load_csv_or_parquet_invalid_extension(tmp_path):
    """Raise ValueError for invalid extension"""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("some data")

    with pytest.raises(ValueError, match="file must be csv or parquet"):
        load_csv_or_parquet(str(txt_file))


# Tests for load_recording


def test_load_recording_csv(tmp_path):
    """Load EEG/EMG arrays from CSV file"""
    csv_filename = str(tmp_path / "recording.csv")
    eeg_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    emg_data = [0.1, 0.2, 0.3, 0.4, 0.5]
    df = pd.DataFrame({c.EEG_COL: eeg_data, c.EMG_COL: emg_data})
    df.to_csv(csv_filename, index=False)

    eeg, emg = load_recording(csv_filename)
    np.testing.assert_array_almost_equal(eeg, eeg_data)
    np.testing.assert_array_almost_equal(emg, emg_data)


def test_load_recording_parquet(tmp_path):
    """Load EEG/EMG arrays from Parquet file"""
    parquet_filename = str(tmp_path / "recording.parquet")
    eeg_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    emg_data = [0.1, 0.2, 0.3, 0.4, 0.5]
    df = pd.DataFrame({c.EEG_COL: eeg_data, c.EMG_COL: emg_data})
    df.to_parquet(parquet_filename, index=False)

    eeg, emg = load_recording(parquet_filename)
    np.testing.assert_array_almost_equal(eeg, eeg_data)
    np.testing.assert_array_almost_equal(emg, emg_data)


# Tests for load_labels


def test_load_labels_without_confidence(tmp_path):
    """Load labels without confidence scores"""
    csv_filename = str(tmp_path / "labels.csv")
    labels = [0, 1, 2, 1, 0]
    df = pd.DataFrame({c.BRAIN_STATE_COL: labels})
    df.to_csv(csv_filename, index=False)

    result_labels, result_confidence = load_labels(csv_filename)
    np.testing.assert_array_equal(result_labels, labels)
    assert result_confidence is None


def test_load_labels_with_confidence(tmp_path):
    """Load labels with confidence scores"""
    csv_filename = str(tmp_path / "labels.csv")
    labels = [0, 1, 2, 1, 0]
    confidence = [0.9, 0.8, 0.9, 0.8, 0.9]
    df = pd.DataFrame({c.BRAIN_STATE_COL: labels, c.CONFIDENCE_SCORE_COL: confidence})
    df.to_csv(csv_filename, index=False)

    result_labels, result_confidence = load_labels(csv_filename)
    np.testing.assert_array_equal(result_labels, labels)
    np.testing.assert_array_almost_equal(result_confidence, confidence)


# Tests for save_labels


def test_save_labels_without_confidence(tmp_path):
    """Save labels without confidence scores"""
    csv_filename = str(tmp_path / "labels.csv")
    labels = np.array([0, 1, 2, 1, 0])

    save_labels(labels, csv_filename)

    df = pd.read_csv(csv_filename)
    assert c.BRAIN_STATE_COL in df.columns
    assert c.CONFIDENCE_SCORE_COL not in df.columns
    np.testing.assert_array_equal(df[c.BRAIN_STATE_COL].values, labels)


def test_save_labels_with_confidence(tmp_path):
    """Save labels with confidence scores"""
    csv_filename = str(tmp_path / "labels.csv")
    labels = np.array([0, 1, 2, 1, 0])
    confidence = np.array([0.9, 0.8, 0.9, 0.8, 0.9])

    save_labels(labels, csv_filename, confidence_scores=confidence)

    df = pd.read_csv(csv_filename)
    assert c.BRAIN_STATE_COL in df.columns
    assert c.CONFIDENCE_SCORE_COL in df.columns
    np.testing.assert_array_equal(df[c.BRAIN_STATE_COL].values, labels)
    np.testing.assert_array_almost_equal(df[c.CONFIDENCE_SCORE_COL].values, confidence)


# Tests for load_calibration_file


def test_load_calibration_file(tmp_path):
    """Load mixture means/SDs correctly"""
    csv_filename = str(tmp_path / "calibration.csv")
    means = [1.0, 2.0, 3.0]
    sds = [0.1, 0.2, 0.3]
    df = pd.DataFrame({c.MIXTURE_MEAN_COL: means, c.MIXTURE_SD_COL: sds})
    df.to_csv(csv_filename, index=False)

    result_means, result_sds = load_calibration_file(csv_filename)
    np.testing.assert_array_almost_equal(result_means, means)
    np.testing.assert_array_almost_equal(result_sds, sds)


# Tests for load_recording_list and save_recording_list


def test_load_recording_list(tmp_path):
    """Load list of recordings from JSON"""
    json_filename = str(tmp_path / "recordings.json")
    recordings_data = {
        c.RECORDING_LIST_NAME: [
            {
                "recording_file": "/path/to/rec1.csv",
                "label_file": "/path/to/labels1.csv",
                "calibration_file": "/path/to/cal1.csv",
                "sampling_rate": 512,
            },
            {
                "recording_file": "/path/to/rec2.csv",
                "label_file": "/path/to/labels2.csv",
                "calibration_file": "/path/to/cal2.csv",
                "sampling_rate": 256,
            },
        ]
    }
    with open(json_filename, "w") as f:
        json.dump(recordings_data, f)

    result = load_recording_list(json_filename)

    assert len(result) == 2
    assert result[0].recording_file == "/path/to/rec1.csv"
    assert result[0].sampling_rate == 512
    assert result[0].name == 1  # name is set to index + 1
    assert result[1].recording_file == "/path/to/rec2.csv"
    assert result[1].sampling_rate == 256
    assert result[1].name == 2


def test_save_recording_list(tmp_path):
    """Save recordings to JSON"""
    json_filename = str(tmp_path / "recordings.json")
    recordings = [
        Recording(
            name=1,
            recording_file="/path/to/rec1.csv",
            label_file="/path/to/labels1.csv",
            calibration_file="/path/to/cal1.csv",
            sampling_rate=512,
        ),
        Recording(
            name=2,
            recording_file="/path/to/rec2.csv",
            label_file="/path/to/labels2.csv",
            calibration_file="/path/to/cal2.csv",
            sampling_rate=256,
        ),
    ]

    save_recording_list(json_filename, recordings)

    with open(json_filename) as f:
        data = json.load(f)

    assert c.RECORDING_LIST_NAME in data
    assert len(data[c.RECORDING_LIST_NAME]) == 2
    assert data[c.RECORDING_LIST_NAME][0]["recording_file"] == "/path/to/rec1.csv"
    assert data[c.RECORDING_LIST_NAME][0]["sampling_rate"] == 512
    assert data[c.RECORDING_LIST_NAME][1]["recording_file"] == "/path/to/rec2.csv"
    assert data[c.RECORDING_LIST_NAME][1]["sampling_rate"] == 256


# Test for get_version


def test_get_version():
    """Return version string from pyproject.toml"""
    version = get_version()
    assert isinstance(version, str)
    # If the file exists, version should not be empty
    # (it might be empty if pyproject.toml doesn't exist in test environment)
