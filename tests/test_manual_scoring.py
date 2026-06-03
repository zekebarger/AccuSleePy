import warnings

import numpy as np
import pytest

from accusleepy.constants import UNDEFINED_LABEL
from accusleepy.gui.manual_scoring import (
    DIGIT_FORMAT,
    DISPLAY_FORMAT,
    ZOOM_FACTOR,
    ZOOM_IN,
    ZOOM_OUT,
    ZOOM_RESET,
    convert_labels,
    find_new_x_limits,
    transform_eeg_emg,
)


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


def test_transform_eeg_emg():
    """EEG and EMG signals are centered and scaled for display"""
    rng = np.random.default_rng(42)
    eeg = rng.normal(5, 1, 10000)
    emg = rng.normal(-3, 2, 10000)

    new_eeg, new_emg = transform_eeg_emg(eeg, emg)

    assert np.mean(new_eeg) == pytest.approx(0)
    assert np.mean(new_emg) == pytest.approx(0)
    assert np.percentile(np.abs(new_eeg), 98) == pytest.approx(1 / 2.2)
    assert np.percentile(np.abs(new_emg), 98) == pytest.approx(1 / 2.2)


def test_transform_eeg_emg_all_zero_emg():
    """A flat (absent) EMG signal stays flat instead of becoming NaN"""
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 1, 10000)
    emg = np.zeros(10000)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        new_eeg, new_emg = transform_eeg_emg(eeg, emg)

    assert np.all(np.isfinite(new_eeg))
    assert np.all(new_emg == 0)
    # EEG should still be scaled normally
    assert np.percentile(np.abs(new_eeg), 98) == pytest.approx(1 / 2.2)


class TestFindNewXLimits:
    """Tests for find_new_x_limits zoom calculation."""

    def test_zoom_reset_returns_full_range(self):
        left, right = find_new_x_limits(
            direction=ZOOM_RESET,
            left_epoch=20,
            right_epoch=40,
            total_epochs=100,
            min_n_shown=5,
            selected_epoch=30,
        )
        assert left == 0
        assert right == 99

    def test_zoom_in_reduces_displayed_range(self):
        left, right = find_new_x_limits(
            direction=ZOOM_IN,
            left_epoch=0,
            right_epoch=99,
            total_epochs=100,
            min_n_shown=5,
            selected_epoch=50,
        )
        n_shown = right - left + 1
        assert n_shown == round(100 * (1 - ZOOM_FACTOR))

    def test_zoom_in_does_not_go_below_min_n_shown(self):
        left, right = find_new_x_limits(
            direction=ZOOM_IN,
            left_epoch=3,
            right_epoch=7,
            total_epochs=100,
            min_n_shown=5,
            selected_epoch=5,
        )
        n_shown = right - left + 1
        assert n_shown >= 5

    def test_zoom_out_increases_displayed_range(self):
        left, right = find_new_x_limits(
            direction=ZOOM_OUT,
            left_epoch=40,
            right_epoch=59,
            total_epochs=100,
            min_n_shown=5,
            selected_epoch=50,
        )
        n_shown = right - left + 1
        assert n_shown == round(20 / (1 - ZOOM_FACTOR))

    def test_zoom_out_does_not_exceed_total_epochs(self):
        left, right = find_new_x_limits(
            direction=ZOOM_OUT,
            left_epoch=0,
            right_epoch=99,
            total_epochs=100,
            min_n_shown=5,
            selected_epoch=50,
        )
        assert left == 0
        assert right == 99

    def test_none_direction_clips_to_min_n_shown(self):
        # current view shows 3 epochs, but min_n_shown is 10
        left, right = find_new_x_limits(
            direction=None,
            left_epoch=48,
            right_epoch=50,
            total_epochs=100,
            min_n_shown=10,
            selected_epoch=49,
        )
        n_shown = right - left + 1
        assert n_shown == 10

    def test_selected_epoch_near_left_boundary(self):
        left, right = find_new_x_limits(
            direction=ZOOM_IN,
            left_epoch=0,
            right_epoch=99,
            total_epochs=100,
            min_n_shown=5,
            selected_epoch=2,
        )
        assert left == 0
        assert right - left + 1 == round(100 * (1 - ZOOM_FACTOR))

    def test_selected_epoch_near_right_boundary(self):
        left, right = find_new_x_limits(
            direction=ZOOM_IN,
            left_epoch=0,
            right_epoch=99,
            total_epochs=100,
            min_n_shown=5,
            selected_epoch=98,
        )
        assert right == 99
        assert right - left + 1 == round(100 * (1 - ZOOM_FACTOR))

    def test_zoom_in_centers_on_selected_epoch(self):
        left, right = find_new_x_limits(
            direction=ZOOM_IN,
            left_epoch=0,
            right_epoch=99,
            total_epochs=100,
            min_n_shown=5,
            selected_epoch=50,
        )
        # selected epoch should be roughly in the middle of the window
        assert left <= 50 <= right
        # check that the distance from selected to each edge differs by at most 1
        assert abs((50 - left) - (right - 50)) <= 1
