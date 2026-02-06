"""Integration tests for the training pipeline."""

import os

import pytest

from accusleepy.constants import DEFAULT_MODEL_TYPE, REAL_TIME_MODEL_TYPE
from accusleepy.models import load_model
from accusleepy.services import TrainingService
from accusleepy.temperature_scaling import ModelWithTemperature


@pytest.mark.integration
class TestTrainingPipeline:
    """Tests for the model training pipeline."""

    def test_train_model_end_to_end(
        self,
        tmp_path,
        synthetic_recording,
        sample_brain_state_set,
        sample_emg_filter,
        fast_hyperparameters,
        epoch_length,
        epochs_per_img,
        calibration_fraction,
    ):
        """Basic training creates a valid model file."""
        model_file = tmp_path / "model.pth"

        service = TrainingService()
        result = service.train_model(
            recordings=[synthetic_recording, synthetic_recording],
            epoch_length=epoch_length,
            epochs_per_img=epochs_per_img,
            model_type=DEFAULT_MODEL_TYPE,
            calibrate=False,
            calibration_fraction=calibration_fraction,
            brain_state_set=sample_brain_state_set,
            emg_filter=sample_emg_filter,
            hyperparameters=fast_hyperparameters,
            model_filename=str(model_file),
            temp_image_dir=str(tmp_path / "images"),
            delete_images=True,
        )

        assert result.success, f"Training failed: {result.error}"
        assert os.path.exists(model_file), "Model file not created"

        # Verify model can be loaded
        (
            model,
            loaded_epoch_length,
            loaded_epochs_per_img,
            model_type,
            brain_states,
        ) = load_model(str(model_file))
        assert loaded_epoch_length == epoch_length
        assert loaded_epochs_per_img == epochs_per_img
        assert model_type == DEFAULT_MODEL_TYPE

        # Verify training images were cleaned up
        assert not os.path.exists(tmp_path / "images")

    def test_train_model_with_calibration(
        self,
        tmp_path,
        synthetic_recording,
        sample_brain_state_set,
        sample_emg_filter,
        fast_hyperparameters,
        epoch_length,
        epochs_per_img,
    ):
        """Training with temperature scaling produces ModelWithTemperature."""
        model_file = tmp_path / "model_calibrated.pth"

        service = TrainingService()
        result = service.train_model(
            recordings=[synthetic_recording],
            epoch_length=epoch_length,
            epochs_per_img=epochs_per_img,
            model_type=DEFAULT_MODEL_TYPE,
            calibrate=True,
            calibration_fraction=0.2,
            brain_state_set=sample_brain_state_set,
            emg_filter=sample_emg_filter,
            hyperparameters=fast_hyperparameters,
            model_filename=str(model_file),
            temp_image_dir=str(tmp_path / "images"),
            delete_images=True,
        )

        assert result.success, f"Training failed: {result.error}"
        assert os.path.exists(model_file)

        # Verify model is calibrated
        model, _, _, _, _ = load_model(str(model_file))
        assert isinstance(model, ModelWithTemperature)

    def test_train_model_real_time_type(
        self,
        tmp_path,
        synthetic_recording,
        sample_brain_state_set,
        sample_emg_filter,
        fast_hyperparameters,
        epoch_length,
        calibration_fraction,
    ):
        """Training real-time model type works."""
        model_file = tmp_path / "model_realtime.pth"

        service = TrainingService()
        result = service.train_model(
            recordings=[synthetic_recording],
            epoch_length=epoch_length,
            epochs_per_img=10,  # Even number allowed for real-time
            model_type=REAL_TIME_MODEL_TYPE,
            calibrate=False,
            calibration_fraction=calibration_fraction,
            brain_state_set=sample_brain_state_set,
            emg_filter=sample_emg_filter,
            hyperparameters=fast_hyperparameters,
            model_filename=str(model_file),
            temp_image_dir=str(tmp_path / "images"),
            delete_images=True,
        )

        assert result.success, f"Training failed: {result.error}"

        model, _, loaded_epochs_per_img, model_type, _ = load_model(str(model_file))
        assert loaded_epochs_per_img == 10
        assert model_type == REAL_TIME_MODEL_TYPE
