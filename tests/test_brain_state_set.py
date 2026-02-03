import numpy as np
import pytest

from accusleepy.brain_state_set import BrainState, BrainStateSet
from accusleepy.constants import UNDEFINED_LABEL

# typical set of brain states
brain_states_default = [
    BrainState(name="REM", digit=1, is_scored=True, frequency=0.1),
    BrainState(name="Wake", digit=2, is_scored=True, frequency=0.35),
    BrainState(name="NREM", digit=3, is_scored=True, frequency=0.55),
]
brain_state_set_default = BrainStateSet(
    brain_states=brain_states_default, undefined_label=UNDEFINED_LABEL
)

# set of brain states with an extra "error" class
brain_states_extra = [
    BrainState(name="REM", digit=1, is_scored=True, frequency=0.1),
    BrainState(name="Wake", digit=2, is_scored=True, frequency=0.35),
    BrainState(name="NREM", digit=3, is_scored=True, frequency=0.55),
    BrainState(name="error", digit=4, is_scored=False, frequency=0),
]
brain_state_set_extra = BrainStateSet(
    brain_states=brain_states_extra, undefined_label=UNDEFINED_LABEL
)

# complicated set of brain states with gaps, unordered
brain_states_gaps = [
    BrainState(name="missing", digit=8, is_scored=False, frequency=0),
    BrainState(name="A", digit=5, is_scored=True, frequency=0.25),  # 0
    BrainState(name="B", digit=0, is_scored=True, frequency=0.25),  # 1
    BrainState(name="C", digit=3, is_scored=True, frequency=0.25),  # 2
    BrainState(name="error", digit=1, is_scored=False, frequency=0),
    BrainState(name="ignore", digit=9, is_scored=False, frequency=0),
    BrainState(name="D", digit=7, is_scored=True, frequency=0.25),  # 3
]
brain_state_set_gaps = BrainStateSet(
    brain_states=brain_states_gaps, undefined_label=UNDEFINED_LABEL
)


def test_undefined_label_used():
    """Undefined label should not be re-used"""
    with pytest.raises(ValueError):
        _ = BrainStateSet(
            brain_states=[
                BrainState(
                    name="A", digit=UNDEFINED_LABEL, is_scored=True, frequency=0.5
                ),
                BrainState(name="B", digit=2, is_scored=True, frequency=0.55),
            ],
            undefined_label=UNDEFINED_LABEL,
        )


def test_default_class_count():
    """Counting classes in the base case"""
    assert brain_state_set_default.n_classes == 3


def test_default_digit_to_class():
    """Digit to class conversion in the base case"""
    assert np.array_equal(
        brain_state_set_default.convert_digit_to_class(
            np.array([UNDEFINED_LABEL, 3, 2, 1])
        ),
        np.array([None, 2, 1, 0]),
    )


def test_default_class_to_digit():
    """Class to digit conversion in the base case"""
    assert np.array_equal(
        brain_state_set_default.convert_class_to_digit(np.array([2, 0, 1])),
        np.array([3, 1, 2]),
    )


def test_extra_class_count():
    """Counting classes with one non-scored digit"""
    assert brain_state_set_extra.n_classes == 3


def test_extra_digit_to_class():
    """Digit to class conversion with one non-scored digit"""
    assert np.array_equal(
        brain_state_set_extra.convert_digit_to_class(
            np.array([UNDEFINED_LABEL, 4, 3, 2, 1])
        ),
        np.array([None, None, 2, 1, 0]),
    )


def test_extra_class_to_digit():
    """Class to digit conversion with one non-scored digit"""
    assert np.array_equal(
        brain_state_set_extra.convert_class_to_digit(np.array([2, 0, 1])),
        np.array([3, 1, 2]),
    )


def test_gaps_class_count():
    """Counting classes with complicated set"""
    assert brain_state_set_gaps.n_classes == 4


def test_gaps_digit_to_class():
    """Digit to class conversion with complicated set"""
    assert np.array_equal(
        brain_state_set_gaps.convert_digit_to_class(
            np.array([UNDEFINED_LABEL, 0, 1, 3, 5, 7, 8, 9])
        ),
        np.array([None, 1, None, 2, 0, 3, None, None]),
    )


def test_gaps_class_to_digit():
    """Class to digit conversion with complicated set"""
    assert np.array_equal(
        brain_state_set_gaps.convert_class_to_digit(np.array([0, 1, 2, 3])),
        np.array([5, 0, 3, 7]),
    )
