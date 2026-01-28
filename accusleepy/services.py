"""Service classes for orchestrating AccuSleePy operations.

Isolating certain functionality here, without any interaction
with UI state, makes it more testable.
"""

import datetime
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from accusleepy.bouts import enforce_min_bout_length
from accusleepy.brain_state_set import BrainStateSet
from accusleepy.constants import (
    ANNOTATIONS_FILENAME,
    CALIBRATION_ANNOTATION_FILENAME,
    DEFAULT_MODEL_TYPE,
    MIN_EPOCHS_PER_STATE,
    UNDEFINED_LABEL,
    MIXTURE_MEAN_COL,
    MIXTURE_SD_COL,
)
from accusleepy.fileio import (
    EMGFilter,
    Hyperparameters,
    Recording,
    load_calibration_file,
    load_labels,
    load_recording,
    save_labels,
)
from accusleepy.models import SSANN
from accusleepy.signal_processing import (
    create_training_images,
    resample_and_standardize,
    create_eeg_emg_image,
    get_mixture_values,
)
from accusleepy.validation import check_label_validity

logger = logging.getLogger(__name__)


@dataclass
class ServiceResult:
    """Result of a service operation."""

    success: bool
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def report_to(self, callback: Callable[[str], None]) -> None:
        """Report all warnings, messages, and errors through a callback.

        :param callback: function to call with each message string
        """
        for warning in self.warnings:
            callback(f"WARNING: {warning}")
        for message in self.messages:
            callback(message)
        if not self.success and self.error:
            callback(f"ERROR: {self.error}")


def check_single_file_inputs(
    recording: Recording,
    epoch_length: int | float,
) -> str | None:
    """Check that a recording's inputs appear valid.

    This runs some basic tests for whether it will be possible to
    load and score a recording. If any test fails, we return an
    error message.

    :param recording: the recording to validate
    :param epoch_length: epoch length in seconds
    :return: error message, or None if valid
    """
    if epoch_length == 0:
        return "epoch length can't be 0"
    if recording.sampling_rate == 0:
        return "sampling rate can't be 0"
    if epoch_length > recording.sampling_rate:
        return "invalid epoch length or sampling rate"
    if recording.recording_file == "":
        return "no recording selected"
    if not os.path.isfile(recording.recording_file):
        return "recording file does not exist"
    if recording.label_file == "":
        return "no label file selected"
    return None


class TrainingService:
    """Service for training classification models."""

    def __init__(self, progress_callback: Callable[[str], None] | None = None) -> None:
        """Initialize the training service.

        :param progress_callback: optional callback for progress messages
        """
        self.progress_callback = progress_callback

    def _report_progress(self, message: str) -> None:
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(message)

    def train_model(
        self,
        recordings: list[Recording],
        epoch_length: int | float,
        epochs_per_img: int,
        model_type: str,
        calibrate: bool,
        calibration_fraction: float,
        brain_state_set: BrainStateSet,
        emg_filter: EMGFilter,
        hyperparameters: Hyperparameters,
        model_filename: str,
        temp_image_dir: str | None = None,
        delete_images: bool = True,
    ) -> ServiceResult:
        """Train a classification model.

        :param recordings: list of recordings to use for training
        :param epoch_length: epoch length in seconds
        :param epochs_per_img: number of epochs per training image
        :param model_type: type of model ('default' or 'real-time')
        :param calibrate: whether to calibrate the model
        :param calibration_fraction: fraction of data to use for calibration
        :param brain_state_set: set of brain state options
        :param emg_filter: EMG filter parameters
        :param hyperparameters: model training hyperparameters
        :param model_filename: path to save the trained model
        :param temp_image_dir: directory for training images (auto-generated if None)
        :param delete_images: whether to delete training images after training
        :return: ServiceResult with status and messages
        """
        result = ServiceResult(success=True)

        # Validate epochs_per_img for default model type
        if model_type == DEFAULT_MODEL_TYPE and epochs_per_img % 2 == 0:
            return ServiceResult(
                success=False,
                error=(
                    "For the default model type, number of epochs "
                    "per image must be an odd number."
                ),
            )

        # Validate each recording
        for recording in recordings:
            error_message = check_single_file_inputs(recording, epoch_length)
            if error_message:
                return ServiceResult(
                    success=False,
                    error=f"Recording {recording.name}: {error_message}",
                )

        # Create temp image directory if not provided
        if temp_image_dir is None:
            temp_image_dir = os.path.join(
                os.path.dirname(model_filename),
                "images_" + datetime.datetime.now().strftime("%Y%m%d%H%M"),
            )

        if os.path.exists(temp_image_dir):
            result.warnings.append("Training image folder exists, will be overwritten")
        os.makedirs(temp_image_dir, exist_ok=True)

        # Create training images
        self._report_progress("Creating training images")
        if not delete_images:
            result.messages.append(f"Creating training images in {temp_image_dir}")
        else:
            result.messages.append(
                f"Creating temporary folder of training images: {temp_image_dir}"
            )

        logger.info("Creating training images")
        failed_recordings, training_class_balance, had_zero_variance = (
            create_training_images(
                recordings=recordings,
                output_path=temp_image_dir,
                epoch_length=epoch_length,
                epochs_per_img=epochs_per_img,
                brain_state_set=brain_state_set,
                model_type=model_type,
                calibration_fraction=calibration_fraction,
                emg_filter=emg_filter,
            )
        )

        if had_zero_variance:
            result.warnings.append(
                "Some recordings contain features with zero variance. "
                "The EEG or EMG signal might be empty. If this is unexpected, "
                "please make sure the recording files are correctly formatted."
            )

        if len(failed_recordings) > 0:
            if len(failed_recordings) == len(recordings):
                # Cleanup before returning error
                if delete_images and os.path.exists(temp_image_dir):
                    shutil.rmtree(temp_image_dir)
                return ServiceResult(
                    success=False,
                    error="No recordings were valid!",
                    warnings=result.warnings,
                )
            else:
                result.warnings.append(
                    "The following recordings could not be loaded and will not "
                    f"be used for training: {', '.join([str(r) for r in failed_recordings])}. "
                    "More information might be available in the terminal."
                )

        # Train model
        self._report_progress("Training model")
        logger.info("Training model")

        from accusleepy.classification import create_dataloader, train_ssann
        from accusleepy.models import save_model
        from accusleepy.temperature_scaling import ModelWithTemperature

        model = train_ssann(
            annotations_file=os.path.join(temp_image_dir, ANNOTATIONS_FILENAME),
            img_dir=temp_image_dir,
            training_class_balance=training_class_balance,
            n_classes=brain_state_set.n_classes,
            hyperparameters=hyperparameters,
        )

        # Calibrate the model if requested
        if calibrate:
            calibration_annotation_file = os.path.join(
                temp_image_dir, CALIBRATION_ANNOTATION_FILENAME
            )
            calibration_dataloader = create_dataloader(
                annotations_file=calibration_annotation_file,
                img_dir=temp_image_dir,
                hyperparameters=hyperparameters,
            )
            model = ModelWithTemperature(model)
            logger.info("Calibrating model")
            model.set_temperature(calibration_dataloader)

        # Save model
        save_model(
            model=model,
            filename=model_filename,
            epoch_length=epoch_length,
            epochs_per_img=epochs_per_img,
            model_type=model_type,
            brain_state_set=brain_state_set,
            is_calibrated=calibrate,
        )

        # Optionally delete images
        if delete_images:
            logger.info("Cleaning up training image folder")
            shutil.rmtree(temp_image_dir)

        result.messages.append(f"Training complete. Saved model to {model_filename}")
        logger.info("Training complete")

        return result


@dataclass
class LoadedModel:
    """State for a loaded classification model."""

    model: SSANN | None = None
    epoch_length: int | float | None = None
    epochs_per_img: int | None = None


def score_recording_list(
    recordings: list[Recording],
    loaded_model: LoadedModel,
    epoch_length: int | float,
    only_overwrite_undefined: bool,
    save_confidence_scores: bool,
    min_bout_length: int | float,
    brain_state_set: BrainStateSet,
    emg_filter: EMGFilter,
) -> ServiceResult:
    """Score all recordings using a classification model.

    :param recordings: list of recordings to score
    :param loaded_model: loaded classification model and metadata
    :param epoch_length: epoch length in seconds
    :param only_overwrite_undefined: only overwrite epochs labeled as undefined
    :param save_confidence_scores: whether to save confidence scores
    :param min_bout_length: minimum bout length in seconds
    :param brain_state_set: set of brain state options
    :param emg_filter: EMG filter parameters
    :return: ServiceResult with status and messages
    """
    result = ServiceResult(success=True)

    # Validate model is loaded
    if loaded_model.model is None:
        return ServiceResult(
            success=False,
            error="No classification model file selected",
        )

    # Validate min_bout_length
    if min_bout_length < epoch_length:
        return ServiceResult(
            success=False,
            error="Minimum bout length must be >= epoch length",
        )

    # Validate model epoch length matches current
    if epoch_length != loaded_model.epoch_length:
        return ServiceResult(
            success=False,
            error=(
                f"Model was trained with an epoch length of "
                f"{loaded_model.epoch_length} seconds, but the current "
                f"epoch length setting is {epoch_length} seconds."
            ),
        )

    # Validate each recording
    for recording in recordings:
        error_message = check_single_file_inputs(recording, epoch_length)
        if error_message:
            return ServiceResult(
                success=False,
                error=f"Recording {recording.name}: {error_message}",
            )
        if recording.calibration_file == "":
            return ServiceResult(
                success=False,
                error=f"Recording {recording.name}: no calibration file selected",
            )

    from accusleepy.classification import score_recording

    any_zero_variance = False

    # Score each recording
    for recording in recordings:
        # Load EEG, EMG
        try:
            eeg, emg = load_recording(recording.recording_file)
            sampling_rate = recording.sampling_rate

            eeg, emg, sampling_rate = resample_and_standardize(
                eeg=eeg,
                emg=emg,
                sampling_rate=sampling_rate,
                epoch_length=epoch_length,
            )
        except Exception:
            logger.exception("Failed to load %s", recording.recording_file)
            result.warnings.append(
                f"Could not load recording {recording.name}. "
                "This recording will be skipped."
            )
            continue

        # Load labels
        label_file = recording.label_file
        if os.path.isfile(label_file):
            try:
                existing_labels, _ = load_labels(label_file)
            except Exception:
                logger.exception("Failed to load %s", label_file)
                result.warnings.append(
                    f"Could not load existing labels for recording "
                    f"{recording.name}. This recording will be skipped."
                )
                continue
            # Only check the length
            samples_per_epoch = sampling_rate * epoch_length
            epochs_in_recording = round(eeg.size / samples_per_epoch)
            if epochs_in_recording != existing_labels.size:
                result.warnings.append(
                    f"Existing labels for recording {recording.name} "
                    "do not match the recording length. "
                    "This recording will be skipped."
                )
                continue
        else:
            existing_labels = None

        # Load calibration data
        if not os.path.isfile(recording.calibration_file):
            result.warnings.append(
                f"Calibration file does not exist for recording "
                f"{recording.name}. This recording will be skipped."
            )
            continue
        try:
            mixture_means, mixture_sds = load_calibration_file(
                recording.calibration_file
            )
        except Exception:
            logger.exception("Failed to load %s", recording.calibration_file)
            result.warnings.append(
                f"Could not load calibration file for recording "
                f"{recording.name}. This recording will be skipped."
            )
            continue

        # Check if calibration data contains any 0-variance features
        if np.any(mixture_sds == 0):
            any_zero_variance = True

        labels, confidence_scores = score_recording(
            model=loaded_model.model,
            eeg=eeg,
            emg=emg,
            mixture_means=mixture_means,
            mixture_sds=mixture_sds,
            sampling_rate=sampling_rate,
            epoch_length=epoch_length,
            epochs_per_img=loaded_model.epochs_per_img,
            brain_state_set=brain_state_set,
            emg_filter=emg_filter,
        )

        # Overwrite as needed
        if existing_labels is not None and only_overwrite_undefined:
            labels[existing_labels != UNDEFINED_LABEL] = existing_labels[
                existing_labels != UNDEFINED_LABEL
            ]

        # Enforce minimum bout length
        labels = enforce_min_bout_length(
            labels=labels,
            epoch_length=epoch_length,
            min_bout_length=min_bout_length,
        )

        # Ignore confidence scores if desired
        if not save_confidence_scores:
            confidence_scores = None

        # Save results
        save_labels(
            labels=labels, filename=label_file, confidence_scores=confidence_scores
        )
        result.messages.append(
            f"Saved labels for recording {recording.name} to {label_file}"
        )

    if any_zero_variance:
        result.warnings.append(
            "One or more calibration files has 0 variance "
            "for some features. This could indicate that the EEG or "
            "EMG signal is empty in the recording used for calibration."
        )

    return result


def create_calibration(
    recording: Recording,
    epoch_length: int | float,
    brain_state_set: BrainStateSet,
    emg_filter: EMGFilter,
    output_filename: str,
) -> ServiceResult:
    """Create a calibration file for a recording.

    :param recording: the recording to create calibration for
    :param epoch_length: epoch length in seconds
    :param brain_state_set: set of brain state options
    :param emg_filter: EMG filter parameters
    :param output_filename: path to save the calibration file
    :return: ServiceResult with status and messages
    """
    result = ServiceResult(success=True)

    # Validate recording inputs
    error_message = check_single_file_inputs(recording, epoch_length)
    if error_message:
        return ServiceResult(success=False, error=error_message)

    # Load the recording
    try:
        eeg, emg = load_recording(recording.recording_file)
    except Exception:
        logger.exception("Failed to load %s", recording.recording_file)
        return ServiceResult(
            success=False,
            error=(
                "Could not load recording. "
                "Check user manual for formatting instructions."
            ),
        )

    sampling_rate = recording.sampling_rate
    eeg, emg, sampling_rate = resample_and_standardize(
        eeg=eeg,
        emg=emg,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
    )

    # Load and validate labels
    label_file = recording.label_file
    if not os.path.isfile(label_file):
        return ServiceResult(
            success=False,
            error="Label file does not exist",
        )

    try:
        labels, _ = load_labels(label_file)
    except Exception:
        logger.exception("Failed to load %s", label_file)
        return ServiceResult(
            success=False,
            error=(
                "Could not load labels. Check user manual for formatting instructions."
            ),
        )

    label_error_message = check_label_validity(
        labels=labels,
        confidence_scores=None,
        samples_in_recording=eeg.size,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        brain_state_set=brain_state_set,
    )
    if label_error_message:
        return ServiceResult(success=False, error=label_error_message)

    # Check that each scored brain state has sufficient observations
    for brain_state in brain_state_set.brain_states:
        if brain_state.is_scored:
            count = np.sum(labels == brain_state.digit)
            if count < MIN_EPOCHS_PER_STATE:
                return ServiceResult(
                    success=False,
                    error=(
                        f"At least {MIN_EPOCHS_PER_STATE} labeled epochs "
                        f"per brain state are required for calibration. Only "
                        f"{count} '{brain_state.name}' epoch(s) found."
                    ),
                )

    # Create calibration file
    img = create_eeg_emg_image(eeg, emg, sampling_rate, epoch_length, emg_filter)
    mixture_means, mixture_sds = get_mixture_values(
        img=img,
        labels=brain_state_set.convert_digit_to_class(labels),
        brain_state_set=brain_state_set,
    )
    pd.DataFrame({MIXTURE_MEAN_COL: mixture_means, MIXTURE_SD_COL: mixture_sds}).to_csv(
        output_filename, index=False
    )

    result.messages.append(
        f"Created calibration file using recording {recording.name} "
        f"at {output_filename}"
    )

    if np.any(mixture_sds == 0):
        result.warnings.append(
            "One or more features derived from the data have "
            "zero variance. This could indicate that the EEG or "
            "EMG signal is empty."
        )

    return result
