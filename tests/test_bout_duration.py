import numpy as np

from accusleepy.signal_processing import enforce_min_bout_length


def test_basic_functionality():
    min_bout_length = 2
    epoch_length = 1
    labels = np.array([1, 1, 2, 2, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3])
    target = np.array([1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)


def test_even_number_sequence():
    min_bout_length = 3
    epoch_length = 1
    labels = np.array([1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    target = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)


def test_odd_number_sequence():
    min_bout_length = 3
    epoch_length = 1
    labels = np.array([1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1])
    target = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )
    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)
