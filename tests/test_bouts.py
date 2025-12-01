import numpy as np
from operator import attrgetter


from accusleepy.bouts import (
    Bout,
    enforce_min_bout_length,
    find_last_adjacent_bout,
    find_short_bouts,
)


def test_convert_labels_to_bouts():
    """Convert labels to bouts correctly"""
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    target_bouts = [
        Bout(length=2, start_index=4, end_index=6, surrounding_state=1),
        Bout(length=2, start_index=2, end_index=4, surrounding_state=0),
        Bout(length=2, start_index=6, end_index=8, surrounding_state=0),
    ]
    bouts = find_short_bouts(labels=labels, min_epochs=3, brain_states={0, 1})
    assert len(bouts) == len(target_bouts)
    for bout, target_bout in zip(bouts, target_bouts):
        assert bout.length == target_bout.length
        assert bout.start_index == target_bout.start_index
        assert bout.end_index == target_bout.end_index
        assert bout.surrounding_state == target_bout.surrounding_state


def test_find_last_adjacent_bout():
    """Find last adjacent bout correctly"""
    # using the same example labels as test_convert_labels_to_bouts
    bouts = [
        Bout(length=2, start_index=4, end_index=6, surrounding_state=1),
        Bout(length=2, start_index=2, end_index=4, surrounding_state=0),
        Bout(length=2, start_index=6, end_index=8, surrounding_state=0),
    ]
    sorted_bouts = sorted(bouts, key=attrgetter("start_index"))
    last_index = find_last_adjacent_bout(sorted_bouts, 0)
    assert last_index == 2


def test_basic_bout_fix():
    """Check if we can remove a bout that's too small"""
    min_bout_length = 2
    epoch_length = 1
    labels = np.array([1, 1, 2, 2, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3])
    target = np.array([1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)


def test_even_number_bout_sequence():
    """Handle two adjacent short bouts"""
    min_bout_length = 3
    epoch_length = 1
    labels = np.array([1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    target = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)


def test_odd_number_bout_sequence():
    """Handle three adjacent short bouts"""
    min_bout_length = 3
    epoch_length = 1
    labels = np.array([1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1])
    target = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    new_labels = enforce_min_bout_length(
        labels=labels, epoch_length=epoch_length, min_bout_length=min_bout_length
    )
    assert np.array_equal(new_labels, target)
