import numpy as np

from accusleepy.brain_state_set import BrainState, BrainStateSet
from accusleepy.constants import UNDEFINED_LABEL
from accusleepy.validation import LABEL_LENGTH_ERROR, check_label_validity

brain_states = [
    BrainState(name="A", digit=0, is_scored=True, frequency=0.5),
    BrainState(name="B", digit=1, is_scored=True, frequency=0.5),
]
brain_state_set = BrainStateSet(
    brain_states=brain_states, undefined_label=UNDEFINED_LABEL
)

sampling_rate = 100  # Hz
epoch_length = 5  # seconds
recording_length = 60  # seconds
samples_in_recording = round(recording_length * sampling_rate)


def test_valid_label_length():
    """Validity check on valid input should return None"""
    labels = np.zeros(round(recording_length / epoch_length))

    output = check_label_validity(
        labels=labels,
        confidence_scores=None,
        samples_in_recording=samples_in_recording,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        brain_state_set=brain_state_set,
    )
    assert output is None


def test_invalid_label_length():
    """Validity check should find invalid label length"""
    labels = np.zeros(10 + round(recording_length / epoch_length))

    output = check_label_validity(
        labels=labels,
        confidence_scores=None,
        samples_in_recording=samples_in_recording,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        brain_state_set=brain_state_set,
    )
    assert output == LABEL_LENGTH_ERROR


def test_invalid_label_entries():
    """Validity check should find invalid labels"""
    labels = np.zeros(round(recording_length / epoch_length))
    labels[2] = 9

    output = check_label_validity(
        labels=labels,
        confidence_scores=None,
        samples_in_recording=samples_in_recording,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        brain_state_set=brain_state_set,
    )
    assert type(output) is str


def test_invalid_confidence_scores():
    """Validity check should find invalid confidence scores"""
    labels = np.zeros(round(recording_length / epoch_length))
    confidence_scores = np.zeros(round(recording_length / epoch_length))
    confidence_scores[5] = -1
    output_below_0 = check_label_validity(
        labels=labels,
        confidence_scores=confidence_scores,
        samples_in_recording=samples_in_recording,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        brain_state_set=brain_state_set,
    )

    confidence_scores[5] = 2
    output_above_1 = check_label_validity(
        labels=labels,
        confidence_scores=confidence_scores,
        samples_in_recording=samples_in_recording,
        sampling_rate=sampling_rate,
        epoch_length=epoch_length,
        brain_state_set=brain_state_set,
    )
    assert type(output_below_0) is str and type(output_above_1) is str
