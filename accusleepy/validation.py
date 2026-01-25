import numpy as np

from accusleepy.brain_state_set import BrainStateSet
from accusleepy.constants import UNDEFINED_LABEL

LABEL_LENGTH_ERROR = "label file length does not match recording length"


def check_label_validity(
    labels: np.ndarray,
    confidence_scores: np.ndarray | None,
    samples_in_recording: int,
    sampling_rate: int | float,
    epoch_length: int | float,
    brain_state_set: BrainStateSet,
) -> str | None:
    """Check whether a set of brain state labels is valid

    This returns an error message if a problem is found with the
    brain state labels.

    :param labels: brain state labels
    :param confidence_scores: confidence scores
    :param samples_in_recording: number of samples in the recording
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :param brain_state_set: BrainStateMapper object
    :return: error message
    """
    # check that number of labels is correct
    samples_per_epoch = round(sampling_rate * epoch_length)
    epochs_in_recording = round(samples_in_recording / samples_per_epoch)
    if epochs_in_recording != labels.size:
        return LABEL_LENGTH_ERROR

    # check that entries are valid
    if not set(labels.tolist()).issubset(
        set([b.digit for b in brain_state_set.brain_states] + [UNDEFINED_LABEL])
    ):
        return "label file contains invalid entries"

    if confidence_scores is not None:
        if np.min(confidence_scores) < 0 or np.max(confidence_scores) > 1:
            return "label file contains invalid confidence scores"

    return None


def check_config_consistency(
    current_brain_states: dict,
    model_brain_states: dict,
    current_epoch_length: int | float,
    model_epoch_length: int | float,
) -> list[str]:
    """Compare current brain state config to the model's config

    This only displays warnings - the user should decide whether to proceed

    :param current_brain_states: current brain state config
    :param model_brain_states: brain state config when the model was created
    :param current_epoch_length: current epoch length setting
    :param model_epoch_length: epoch length used when the model was created
    """
    output = list()

    # make lists of names and digits for scored brain states
    current_scored_states = {
        f: [b[f] for b in current_brain_states if b["is_scored"]]
        for f in ["name", "digit"]
    }
    model_scored_states = {
        f: [b[f] for b in model_brain_states if b["is_scored"]]
        for f in ["name", "digit"]
    }

    # generate message comparing the brain state configs
    config_comparisons = list()
    for config, config_name in zip(
        [current_scored_states, model_scored_states], ["current", "model's"]
    ):
        config_comparisons.append(
            f"Scored brain states in {config_name} configuration: "
            f"""{
                ", ".join(
                    [
                        f"{x}: {y}"
                        for x, y in zip(
                            config["digit"],
                            config["name"],
                        )
                    ]
                )
            }"""
        )

    # check if the number of scored states is different
    len_diff = len(current_scored_states["name"]) - len(model_scored_states["name"])
    if len_diff != 0:
        output.append(
            (
                "WARNING: current brain state configuration has "
                f"{'fewer' if len_diff < 0 else 'more'} "
                "scored brain states than the model's configuration."
            )
        )
        output = output + config_comparisons
    else:
        # the length is the same, but names might be different
        if current_scored_states["name"] != model_scored_states["name"]:
            output.append(
                (
                    "WARNING: current brain state configuration appears "
                    "to contain different brain states than "
                    "the model's configuration."
                )
            )
            output = output + config_comparisons

    if current_epoch_length != model_epoch_length:
        output.append(
            (
                "Warning: the epoch length used when training this model "
                f"({model_epoch_length} seconds) "
                "does not match the current epoch length setting."
            )
        )

    return output
