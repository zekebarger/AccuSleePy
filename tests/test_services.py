"""Tests for service classes and functions in accusleepy/services.py."""

import os
import tempfile

import numpy as np
import pytest

from accusleepy.brain_state_set import BrainState, BrainStateSet
from accusleepy.constants import DEFAULT_MODEL_TYPE, UNDEFINED_LABEL
from accusleepy.fileio import EMGFilter, Hyperparameters, Recording
from accusleepy.models import SSANN
from accusleepy.services import (
    LoadedModel,
    ServiceResult,
    TrainingService,
    check_single_file_inputs,
    create_calibration,
    score_recording_list,
)


# Test fixtures
@pytest.fixture
def brain_state_set():
    """Create a basic brain state set for testing."""
    brain_states = [
        BrainState(name="Wake", digit=0, is_scored=True, frequency=0.33),
        BrainState(name="NREM", digit=1, is_scored=True, frequency=0.34),
        BrainState(name="REM", digit=2, is_scored=True, frequency=0.33),
    ]
    return BrainStateSet(brain_states=brain_states, undefined_label=UNDEFINED_LABEL)


@pytest.fixture
def emg_filter():
    """Create default EMG filter settings."""
    return EMGFilter(order=8, bp_lower=20, bp_upper=50)


@pytest.fixture
def hyperparameters():
    """Create default hyperparameters."""
    return Hyperparameters(
        batch_size=64, learning_rate=1e-3, momentum=0.9, training_epochs=6
    )


@pytest.fixture
def temp_recording_file():
    """Create a temporary recording file with valid data."""
    import pandas as pd

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Create minimal valid recording data
        sampling_rate = 100
        duration = 60  # seconds
        n_samples = sampling_rate * duration
        eeg = np.random.randn(n_samples)
        emg = np.random.randn(n_samples)
        df = pd.DataFrame({"eeg": eeg, "emg": emg})
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_label_file(brain_state_set):
    """Create a temporary label file with valid labels."""
    import pandas as pd

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # 60 seconds / 5 second epochs = 12 epochs
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        df = pd.DataFrame({"brain_state": labels})
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def valid_recording(temp_recording_file, temp_label_file):
    """Create a valid Recording object."""
    return Recording(
        name=1,
        recording_file=temp_recording_file,
        label_file=temp_label_file,
        calibration_file="",
        sampling_rate=100,
    )


@pytest.fixture
def dummy_model():
    """Create a dummy SSANN model for testing."""
    return SSANN(n_classes=3)


class TestServiceResult:
    """Tests for ServiceResult dataclass."""

    def test_default_values(self):
        """ServiceResult should have sensible defaults."""
        result = ServiceResult(success=True)
        assert result.success is True
        assert result.messages == []
        assert result.warnings == []
        assert result.error is None

    def test_failure_result(self):
        """ServiceResult can represent failures."""
        result = ServiceResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_with_messages_and_warnings(self):
        """ServiceResult can hold messages and warnings."""
        result = ServiceResult(
            success=True,
            messages=["Task completed", "File saved"],
            warnings=["Data will be overwritten"],
        )
        assert len(result.messages) == 2
        assert len(result.warnings) == 1

    def test_report_to_success(self):
        """report_to should output warnings and messages for successful result."""
        result = ServiceResult(
            success=True,
            messages=["File saved"],
            warnings=["Data will be overwritten"],
        )
        output = []
        result.report_to(output.append)
        assert output == ["WARNING: Data will be overwritten", "File saved"]

    def test_report_to_failure(self):
        """report_to should output error for failed result."""
        result = ServiceResult(
            success=False,
            messages=["Partial progress"],
            warnings=["Warning 1"],
            error="Something failed",
        )
        output = []
        result.report_to(output.append)
        assert output == [
            "WARNING: Warning 1",
            "Partial progress",
            "ERROR: Something failed",
        ]

    def test_report_to_empty(self):
        """report_to with no messages should produce no output."""
        result = ServiceResult(success=True)
        output = []
        result.report_to(output.append)
        assert output == []


class TestCheckSingleFileInputs:
    """Tests for check_single_file_inputs validation function."""

    def test_valid_recording(self, valid_recording):
        """Valid recording should return None."""
        result = check_single_file_inputs(valid_recording, epoch_length=5)
        assert result is None

    def test_zero_epoch_length(self, valid_recording):
        """Zero epoch length should return error."""
        result = check_single_file_inputs(valid_recording, epoch_length=0)
        assert result == "epoch length can't be 0"

    def test_zero_sampling_rate(self, temp_recording_file, temp_label_file):
        """Zero sampling rate should return error."""
        recording = Recording(
            recording_file=temp_recording_file,
            label_file=temp_label_file,
            sampling_rate=0,
        )
        result = check_single_file_inputs(recording, epoch_length=5)
        assert result == "sampling rate can't be 0"

    def test_epoch_length_exceeds_sampling_rate(
        self, temp_recording_file, temp_label_file
    ):
        """Epoch length > sampling rate should return error."""
        recording = Recording(
            recording_file=temp_recording_file,
            label_file=temp_label_file,
            sampling_rate=2,
        )
        result = check_single_file_inputs(recording, epoch_length=5)
        assert result == "invalid epoch length or sampling rate"

    def test_no_recording_file(self, temp_label_file):
        """Empty recording file should return error."""
        recording = Recording(
            recording_file="",
            label_file=temp_label_file,
            sampling_rate=100,
        )
        result = check_single_file_inputs(recording, epoch_length=5)
        assert result == "no recording selected"

    def test_nonexistent_recording_file(self, temp_label_file):
        """Nonexistent recording file should return error."""
        recording = Recording(
            recording_file="/nonexistent/path/recording.csv",
            label_file=temp_label_file,
            sampling_rate=100,
        )
        result = check_single_file_inputs(recording, epoch_length=5)
        assert result == "recording file does not exist"

    def test_no_label_file(self, temp_recording_file):
        """Empty label file should return error."""
        recording = Recording(
            recording_file=temp_recording_file,
            label_file="",
            sampling_rate=100,
        )
        result = check_single_file_inputs(recording, epoch_length=5)
        assert result == "no label file selected"


class TestTrainingService:
    """Tests for TrainingService class."""

    def test_init_with_callback(self):
        """TrainingService accepts a progress callback."""
        messages = []
        service = TrainingService(progress_callback=messages.append)
        service._report_progress("Test message")
        assert "Test message" in messages

    def test_init_without_callback(self):
        """TrainingService works without a callback."""
        service = TrainingService()
        # Should not raise
        service._report_progress("Test message")

    def test_even_epochs_per_img_default_model(
        self, valid_recording, brain_state_set, emg_filter, hyperparameters
    ):
        """Even epochs_per_img should fail for default model type."""
        service = TrainingService()
        result = service.train_model(
            recordings=[valid_recording],
            epoch_length=5,
            epochs_per_img=8,  # Even number - should fail
            model_type=DEFAULT_MODEL_TYPE,
            calibrate=False,
            calibration_fraction=0,
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
            hyperparameters=hyperparameters,
            model_filename="/tmp/test_model.pth",
            delete_images=True,
        )
        assert result.success is False
        assert "odd number" in result.error.lower()

    def test_invalid_recording_fails(
        self, brain_state_set, emg_filter, hyperparameters
    ):
        """Invalid recording should fail validation."""
        invalid_recording = Recording(
            recording_file="/nonexistent/file.csv",
            label_file="/also/nonexistent.csv",
            sampling_rate=100,
        )
        service = TrainingService()
        result = service.train_model(
            recordings=[invalid_recording],
            epoch_length=5,
            epochs_per_img=9,
            model_type=DEFAULT_MODEL_TYPE,
            calibrate=False,
            calibration_fraction=0,
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
            hyperparameters=hyperparameters,
            model_filename="/tmp/test_model.pth",
            delete_images=True,
        )
        assert result.success is False
        assert "does not exist" in result.error.lower()


class TestScoreRecordingList:
    """Tests for score_recording_list function."""

    def test_no_model_loaded(self, valid_recording, brain_state_set, emg_filter):
        """Scoring with no model should fail."""
        loaded_model = LoadedModel(model=None)
        result = score_recording_list(
            recordings=[valid_recording],
            loaded_model=loaded_model,
            epoch_length=5,
            only_overwrite_undefined=True,
            save_confidence_scores=True,
            min_bout_length=5,
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
        )
        assert result.success is False
        assert "no classification model" in result.error.lower()

    def test_min_bout_length_too_small(
        self, valid_recording, brain_state_set, emg_filter, dummy_model
    ):
        """Min bout length < epoch length should fail."""
        loaded_model = LoadedModel(
            model=dummy_model,
            epoch_length=5,
            epochs_per_img=9,
        )
        result = score_recording_list(
            recordings=[valid_recording],
            loaded_model=loaded_model,
            epoch_length=5,
            only_overwrite_undefined=True,
            save_confidence_scores=True,
            min_bout_length=3,  # Less than epoch_length of 5
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
        )
        assert result.success is False
        assert "minimum bout length" in result.error.lower()

    def test_epoch_length_mismatch(
        self, valid_recording, brain_state_set, emg_filter, dummy_model
    ):
        """Epoch length mismatch with model should fail."""
        loaded_model = LoadedModel(
            model=dummy_model,
            epoch_length=10,  # Different from current epoch_length
            epochs_per_img=9,
        )
        result = score_recording_list(
            recordings=[valid_recording],
            loaded_model=loaded_model,
            epoch_length=5,  # Different from model's epoch_length
            only_overwrite_undefined=True,
            save_confidence_scores=True,
            min_bout_length=5,
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
        )
        assert result.success is False
        assert "epoch length" in result.error.lower()

    def test_missing_calibration_file(
        self, valid_recording, brain_state_set, emg_filter, dummy_model
    ):
        """Recording without calibration file should fail."""
        loaded_model = LoadedModel(
            model=dummy_model,
            epoch_length=5,
            epochs_per_img=9,
        )
        # valid_recording has no calibration file by default
        result = score_recording_list(
            recordings=[valid_recording],
            loaded_model=loaded_model,
            epoch_length=5,
            only_overwrite_undefined=True,
            save_confidence_scores=True,
            min_bout_length=5,
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
        )
        assert result.success is False
        assert "calibration file" in result.error.lower()


class TestCreateCalibration:
    """Tests for create_calibration function."""

    def test_invalid_recording_fails(self, brain_state_set, emg_filter):
        """Invalid recording should fail validation."""
        invalid_recording = Recording(
            recording_file="/nonexistent/file.csv",
            label_file="/also/nonexistent.csv",
            sampling_rate=100,
        )
        result = create_calibration(
            recording=invalid_recording,
            epoch_length=5,
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
            output_filename="/tmp/calibration.csv",
        )
        assert result.success is False
        assert "does not exist" in result.error.lower()

    def test_missing_label_file(self, temp_recording_file, brain_state_set, emg_filter):
        """Missing label file should fail."""
        recording = Recording(
            recording_file=temp_recording_file,
            label_file="/nonexistent/labels.csv",
            sampling_rate=100,
        )
        result = create_calibration(
            recording=recording,
            epoch_length=5,
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
            output_filename="/tmp/calibration.csv",
        )
        assert result.success is False
        # Should fail on label file not existing
        assert (
            "label file" in result.error.lower()
            or "does not exist" in result.error.lower()
        )


class TestLoadedModel:
    """Tests for LoadedModel dataclass."""

    def test_default_values(self):
        """LoadedModel should have None defaults."""
        model = LoadedModel()
        assert model.model is None
        assert model.epoch_length is None
        assert model.epochs_per_img is None

    def test_with_values(self, dummy_model):
        """LoadedModel can store model info."""
        loaded = LoadedModel(
            model=dummy_model,
            epoch_length=5,
            epochs_per_img=9,
        )
        assert loaded.model is dummy_model
        assert loaded.epoch_length == 5
        assert loaded.epochs_per_img == 9
