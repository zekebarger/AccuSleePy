"""Integration tests for the scoring pipeline."""

import numpy as np
import pandas as pd
import pytest

from accusleepy.constants import (
    BRAIN_STATE_COL,
    DEFAULT_MODEL_TYPE,
    UNDEFINED_LABEL,
)
from accusleepy.fileio import Recording, load_labels
from accusleepy.models import load_model
from accusleepy.services import (
    LoadedModel,
    TrainingService,
    create_calibration,
    score_recording_list,
)


@pytest.fixture
def trained_model_and_calibration(
    tmp_path,
    synthetic_recording,
    sample_brain_state_set,
    sample_emg_filter,
    fast_hyperparameters,
):
    """Create a trained model and calibration file for scoring tests."""
    # Train model
    model_file = tmp_path / "model.pth"
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

    # Create calibration file
    calibration_file = tmp_path / "calibration.csv"
    cal_result = create_calibration(
        recording=synthetic_recording,
        epoch_length=4,
        brain_state_set=sample_brain_state_set,
        emg_filter=sample_emg_filter,
        output_filename=str(calibration_file),
    )
    assert cal_result.success, f"Calibration failed: {cal_result.error}"

    return {
        "model_file": model_file,
        "calibration_file": calibration_file,
        "recording": synthetic_recording,
    }


@pytest.mark.integration
class TestScoringPipeline:
    """Tests for the automated scoring pipeline."""

    def test_score_recording_end_to_end(
        self,
        synthetic_recording,
        synthetic_recording_data,
        sample_brain_state_set,
        sample_emg_filter,
        trained_model_and_calibration,
    ):
        """Full scoring produces valid labels and confidence scores."""
        # Load model
        model, epoch_length, epochs_per_img, _, _ = load_model(
            str(trained_model_and_calibration["model_file"])
        )
        loaded_model = LoadedModel(
            model=model,
            epoch_length=epoch_length,
            epochs_per_img=epochs_per_img,
        )

        # Score the recording
        result = score_recording_list(
            recordings=[synthetic_recording],
            loaded_model=loaded_model,
            epoch_length=4,
            only_overwrite_undefined=False,
            save_confidence_scores=True,
            min_bout_length=4,
            brain_state_set=sample_brain_state_set,
            emg_filter=sample_emg_filter,
        )

        assert result.success, f"Scoring failed: {result.error}"

        # Verify scored labels
        n_epochs = synthetic_recording_data["n_epochs"]
        labels, confidence_scores = load_labels(synthetic_recording.label_file)

        assert len(labels) == n_epochs, "Wrong number of labels"
        assert all(label in [0, 1, 2] for label in labels), "Invalid label values"

        assert confidence_scores is not None, "Confidence scores not saved"
        assert all(0 <= c <= 1 for c in confidence_scores), (
            "Invalid confidence score range"
        )

    def test_score_and_only_overwrite_undefined(
        self,
        tmp_path,
        synthetic_recording_file,
        synthetic_recording_data,
        sample_brain_state_set,
        sample_emg_filter,
        trained_model_and_calibration,
    ):
        """only_overwrite_undefined=True preserves existing labels."""
        # Create label file with some existing labels and some undefined
        # The existing labels will have invalid states to prove that they
        #    are not overwritten
        n_epochs = synthetic_recording_data["n_epochs"]
        existing_labels = np.array([UNDEFINED_LABEL] * n_epochs)
        existing_labels[:5] = [9, 9, 9, 9, 9]

        label_file = tmp_path / "partial_labels.csv"
        pd.DataFrame({BRAIN_STATE_COL: existing_labels}).to_csv(label_file, index=False)

        test_recording = Recording(
            name=1,
            recording_file=str(synthetic_recording_file),
            label_file=str(label_file),
            calibration_file=str(trained_model_and_calibration["calibration_file"]),
            sampling_rate=synthetic_recording_data["sampling_rate"],
        )

        # Load model
        model, epoch_length, epochs_per_img, _, _ = load_model(
            str(trained_model_and_calibration["model_file"])
        )
        loaded_model = LoadedModel(
            model=model,
            epoch_length=epoch_length,
            epochs_per_img=epochs_per_img,
        )

        # Score with only_overwrite_undefined=True
        result = score_recording_list(
            recordings=[test_recording],
            loaded_model=loaded_model,
            epoch_length=4,
            only_overwrite_undefined=True,
            save_confidence_scores=False,
            min_bout_length=4,
            brain_state_set=sample_brain_state_set,
            emg_filter=sample_emg_filter,
        )

        assert result.success, f"Scoring failed: {result.error}"

        # Verify existing labels were preserved
        labels, _ = load_labels(str(label_file))

        expected_first_5 = [9, 9, 9, 9, 9]
        assert list(labels[:5]) == expected_first_5, "Existing labels were overwritten"

        # Remaining labels should no longer be undefined
        assert all(label != UNDEFINED_LABEL for label in labels[5:]), (
            "Undefined labels not scored"
        )
