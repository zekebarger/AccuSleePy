import numpy as np
import pytest

from accusleepy.brain_state_set import BRAIN_STATES_KEY, BrainState, BrainStateSet
from accusleepy.constants import UNDEFINED_LABEL
from accusleepy.validation import (
    LABEL_LENGTH_ERROR,
    check_config_consistency,
    check_label_validity,
    validate_and_correct_labels,
)

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


@pytest.fixture
def brain_state_set_3_states():
    """Create a brain state set with 3 states for testing."""
    states = [
        BrainState(name="REM", digit=0, is_scored=True, frequency=0.1),
        BrainState(name="Wake", digit=1, is_scored=True, frequency=0.35),
        BrainState(name="NREM", digit=2, is_scored=True, frequency=0.55),
    ]
    return BrainStateSet(brain_states=states, undefined_label=UNDEFINED_LABEL)


def test_check_config_consistency():
    """Config consistency check works as expected"""
    # two states: A (0) and B (1)
    brain_state_dict_1 = brain_state_set.to_output_dict()[BRAIN_STATES_KEY]

    # more classes
    brain_states_2 = [
        BrainState(name="A", digit=0, is_scored=True, frequency=0.5),
        BrainState(name="B", digit=1, is_scored=True, frequency=0.25),
        BrainState(name="C", digit=2, is_scored=True, frequency=0.25),
    ]
    brain_state_dict_2 = BrainStateSet(
        brain_states=brain_states_2, undefined_label=UNDEFINED_LABEL
    ).to_output_dict()[BRAIN_STATES_KEY]

    # different state names
    # Unless brain states are manually modified in the config,
    # they will always be ordered by digit. So, a different order
    # means that the states have different digits.
    brain_states_3 = [
        BrainState(name="B", digit=0, is_scored=True, frequency=0.5),
        BrainState(name="A", digit=1, is_scored=True, frequency=0.5),
    ]
    brain_state_dict_3 = BrainStateSet(
        brain_states=brain_states_3, undefined_label=UNDEFINED_LABEL
    ).to_output_dict()[BRAIN_STATES_KEY]

    # extra unscored class that can be ignored
    brain_states_4 = [
        BrainState(name="A", digit=0, is_scored=True, frequency=0.5),
        BrainState(name="x", digit=1, is_scored=False, frequency=0),
        BrainState(name="B", digit=2, is_scored=True, frequency=0.5),
    ]
    brain_state_dict_4 = BrainStateSet(
        brain_states=brain_states_4, undefined_label=UNDEFINED_LABEL
    ).to_output_dict()[BRAIN_STATES_KEY]

    output_2 = check_config_consistency(
        current_brain_states=brain_state_dict_1,
        model_brain_states=brain_state_dict_2,
        current_epoch_length=5,
        model_epoch_length=5,
    )
    output_3 = check_config_consistency(
        current_brain_states=brain_state_dict_1,
        model_brain_states=brain_state_dict_3,
        current_epoch_length=5,
        model_epoch_length=5,
    )
    output_4 = check_config_consistency(
        current_brain_states=brain_state_dict_1,
        model_brain_states=brain_state_dict_4,
        current_epoch_length=5,
        model_epoch_length=5,
    )

    assert len(output_2) > 0
    assert len(output_3) > 0
    assert len(output_4) == 0


class TestValidateAndCorrectLabels:
    """Tests for validate_and_correct_labels function."""

    def test_valid_labels(self, brain_state_set_3_states):
        """Valid labels should return success with no changes."""
        # 12 epochs at 5 seconds each = 60 seconds
        # 60 seconds * 100 Hz = 6000 samples
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        result_labels, result_scores, message = validate_and_correct_labels(
            labels=labels,
            confidence_scores=None,
            samples_in_recording=6000,
            sampling_rate=100,
            epoch_length=5,
            brain_state_set=brain_state_set_3_states,
        )
        assert result_labels is not None
        assert message is None
        assert np.array_equal(result_labels, labels)

    def test_labels_one_short_gets_padded(self, brain_state_set_3_states):
        """Labels that are one epoch short should be padded."""
        # 11 labels but recording has 12 epochs worth of samples
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        result_labels, result_scores, message = validate_and_correct_labels(
            labels=labels,
            confidence_scores=None,
            samples_in_recording=6000,
            sampling_rate=100,
            epoch_length=5,
            brain_state_set=brain_state_set_3_states,
        )
        assert result_labels is not None
        assert message is not None
        assert "added" in message.lower()
        assert result_labels.size == 12
        assert result_labels[-1] == UNDEFINED_LABEL

    def test_labels_one_long_gets_truncated(self, brain_state_set_3_states):
        """Labels that are one epoch long should be truncated."""
        # 13 labels but recording has 12 epochs worth of samples
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        result_labels, result_scores, message = validate_and_correct_labels(
            labels=labels,
            confidence_scores=None,
            samples_in_recording=6000,
            sampling_rate=100,
            epoch_length=5,
            brain_state_set=brain_state_set_3_states,
        )
        assert result_labels is not None
        assert message is not None
        assert "removed" in message.lower()
        assert result_labels.size == 12

    def test_labels_way_off_fails(self, brain_state_set_3_states):
        """Labels with a big length mismatch should fail validation."""
        # 5 labels but recording has 12 epochs worth of samples
        labels = np.array([0, 1, 2, 0, 1])
        result_labels, result_scores, message = validate_and_correct_labels(
            labels=labels,
            confidence_scores=None,
            samples_in_recording=6000,
            sampling_rate=100,
            epoch_length=5,
            brain_state_set=brain_state_set_3_states,
        )
        assert result_labels is None
        assert message is not None  # Error message

    def test_invalid_label_value_fails(self, brain_state_set_3_states):
        """Invalid label values should fail validation."""
        labels = np.array([0, 1, 2, 0, 1, 99, 0, 1, 2, 0, 1, 2])  # 99 is invalid
        result_labels, result_scores, message = validate_and_correct_labels(
            labels=labels,
            confidence_scores=None,
            samples_in_recording=6000,
            sampling_rate=100,
            epoch_length=5,
            brain_state_set=brain_state_set_3_states,
        )
        assert result_labels is None
        assert message is not None  # Error message

    def test_confidence_scores_padded_with_labels(self, brain_state_set_3_states):
        """Confidence scores should be padded when labels are padded."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        confidence_scores = np.array([0.9] * 11)
        result_labels, result_scores, message = validate_and_correct_labels(
            labels=labels,
            confidence_scores=confidence_scores,
            samples_in_recording=6000,
            sampling_rate=100,
            epoch_length=5,
            brain_state_set=brain_state_set_3_states,
        )
        assert result_labels is not None
        assert result_scores.size == 12
        assert result_scores[-1] == 0  # Padded with 0

    def test_confidence_scores_truncated_with_labels(self, brain_state_set_3_states):
        """Confidence scores should be truncated when labels are truncated."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        confidence_scores = np.array([0.9] * 13)
        result_labels, result_scores, message = validate_and_correct_labels(
            labels=labels,
            confidence_scores=confidence_scores,
            samples_in_recording=6000,
            sampling_rate=100,
            epoch_length=5,
            brain_state_set=brain_state_set_3_states,
        )
        assert result_labels is not None
        assert result_scores.size == 12
