import numpy as np

from accusleepy.constants import UNDEFINED_LABEL
from accusleepy.gui.manual_scoring import DIGIT_FORMAT, DISPLAY_FORMAT, convert_labels


def test_digit_to_display():
    labels = np.array([0, 1, 2, 3, UNDEFINED_LABEL])
    target = np.array([10, 1, 2, 3, 0])
    assert np.array_equal(convert_labels(labels, DISPLAY_FORMAT), target)


def test_display_to_digit():
    labels = np.array([10, 1, 2, 3, 0])
    target = np.array([0, 1, 2, 3, UNDEFINED_LABEL])
    assert np.array_equal(convert_labels(labels, DIGIT_FORMAT), target)


def test_convert_labels():
    digit_labels = np.array([0, 1, 2, 3, UNDEFINED_LABEL, 2, 1, 0])
    display_labels = np.array([10, 1, 2, 3, 0, 2, 1, 10])

    assert np.array_equal(convert_labels(digit_labels, DISPLAY_FORMAT), display_labels)
    assert np.array_equal(convert_labels(display_labels, DIGIT_FORMAT), digit_labels)
