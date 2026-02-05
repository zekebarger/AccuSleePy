"""Integration tests for GUI components with real data."""

import pytest

from accusleepy.constants import DEFAULT_MODEL_TYPE
from accusleepy.gui.main import AccuSleepWindow
from accusleepy.services import TrainingService


@pytest.fixture
def trained_model_file(
    tmp_path,
    synthetic_recording,
    sample_brain_state_set,
    sample_emg_filter,
    fast_hyperparameters,
):
    """Create a trained model file for testing."""
    model_file = tmp_path / "test_model.pth"
    service = TrainingService()
    result = service.train_model(
        recordings=[synthetic_recording],
        epoch_length=4,
        epochs_per_img=9,
        model_type=DEFAULT_MODEL_TYPE,
        calibrate=False,
        calibration_fraction=0.2,
        brain_state_set=sample_brain_state_set,
        emg_filter=sample_emg_filter,
        hyperparameters=fast_hyperparameters,
        model_filename=str(model_file),
        temp_image_dir=str(tmp_path / "images"),
        delete_images=True,
    )
    assert result.success, f"Training failed: {result.error}"

    return model_file


@pytest.mark.gui
@pytest.mark.integration
class TestMainWindowIntegration:
    """Integration tests for AccuSleepWindow with real models."""

    def test_main_window_load_model(self, qtbot, trained_model_file):
        """AccuSleepWindow loads trained model correctly."""
        window = AccuSleepWindow()
        qtbot.addWidget(window)

        assert window.loaded_model.model is None

        # Load the trained model
        window.load_model(filename=str(trained_model_file))

        # Verify model was loaded
        assert window.loaded_model.model is not None
        assert window.loaded_model.epoch_length == 4
        assert window.loaded_model.epochs_per_img == 9

        # Verify UI was updated
        assert str(trained_model_file) in window.ui.model_label.text()
