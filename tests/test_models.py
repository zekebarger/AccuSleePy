"""Tests for accusleepy/models.py"""

import torch

from accusleepy.constants import IMAGE_HEIGHT
from accusleepy.models import SSANN, load_model, save_model
from accusleepy.temperature_scaling import ModelWithTemperature


class TestSSANN:
    """Tests for the SSANN neural network"""

    def test_init_creates_correct_layers(self):
        """Model instantiates with correct layers"""
        n_classes = 3
        model = SSANN(n_classes=n_classes)

        # Check that all expected layers exist
        assert hasattr(model, "pool")
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "conv3")
        assert hasattr(model, "conv1_bn")
        assert hasattr(model, "conv2_bn")
        assert hasattr(model, "conv3_bn")
        assert hasattr(model, "fc1")

        # Check final layer output size
        assert model.fc1.out_features == n_classes

    def test_forward_produces_correct_output_shape(self):
        """Forward pass produces correct output shape"""
        n_classes = 3
        batch_size = 4
        epochs_per_img = 9

        model = SSANN(n_classes=n_classes)
        model.eval()

        # Create input tensor with shape (batch, channels, height, width)
        # Height is IMAGE_HEIGHT, width is epochs_per_img
        x = torch.randn(batch_size, 1, IMAGE_HEIGHT, epochs_per_img)

        with torch.no_grad():
            output = model(x)

        # Output should be (batch_size, n_classes)
        assert output.shape == (batch_size, n_classes)


class TestSaveLoadModel:
    """Tests for model saving and loading"""

    def test_save_model(self, tmp_path, sample_brain_state_set):
        """Model saves to file with metadata"""
        n_classes = 3
        model = SSANN(n_classes=n_classes)
        filename = str(tmp_path / "test_model.pth")
        epoch_length = 5
        epochs_per_img = 9
        model_type = "default"

        save_model(
            model=model,
            filename=filename,
            epoch_length=epoch_length,
            epochs_per_img=epochs_per_img,
            model_type=model_type,
            brain_state_set=sample_brain_state_set,
            is_calibrated=False,
        )

        # Check file was created
        assert (tmp_path / "test_model.pth").exists()

        # Load and verify metadata
        state_dict = torch.load(filename, weights_only=True)
        assert state_dict["epoch_length"] == epoch_length
        assert state_dict["epochs_per_img"] == epochs_per_img
        assert state_dict["model_type"] == model_type
        assert state_dict["is_calibrated"] is False

    def test_load_model_non_calibrated(self, tmp_path, sample_brain_state_set):
        """Non-calibrated model loads as SSANN"""
        n_classes = 3
        model = SSANN(n_classes=n_classes)
        filename = str(tmp_path / "test_model.pth")
        epoch_length = 5
        epochs_per_img = 9
        model_type = "default"

        save_model(
            model=model,
            filename=filename,
            epoch_length=epoch_length,
            epochs_per_img=epochs_per_img,
            model_type=model_type,
            brain_state_set=sample_brain_state_set,
            is_calibrated=False,
        )

        (
            loaded_model,
            loaded_epoch_length,
            loaded_epochs_per_img,
            loaded_model_type,
            loaded_brain_states,
        ) = load_model(filename)

        assert isinstance(loaded_model, SSANN)
        assert not isinstance(loaded_model, ModelWithTemperature)
        assert loaded_epoch_length == epoch_length
        assert loaded_epochs_per_img == epochs_per_img
        assert loaded_model_type == model_type
        assert len(loaded_brain_states) == 3

    def test_load_model_calibrated(self, tmp_path, sample_brain_state_set):
        """Calibrated model loads as ModelWithTemperature"""
        n_classes = 3
        base_model = SSANN(n_classes=n_classes)
        model = ModelWithTemperature(base_model)
        filename = str(tmp_path / "test_model.pth")
        epoch_length = 5
        epochs_per_img = 9
        model_type = "default"

        save_model(
            model=model,
            filename=filename,
            epoch_length=epoch_length,
            epochs_per_img=epochs_per_img,
            model_type=model_type,
            brain_state_set=sample_brain_state_set,
            is_calibrated=True,
        )

        (
            loaded_model,
            loaded_epoch_length,
            loaded_epochs_per_img,
            loaded_model_type,
            _loaded_brain_states,
        ) = load_model(filename)

        assert isinstance(loaded_model, ModelWithTemperature)
        assert loaded_epoch_length == epoch_length
        assert loaded_epochs_per_img == epochs_per_img
        assert loaded_model_type == model_type
