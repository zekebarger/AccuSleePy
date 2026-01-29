"""Shared fixtures for AccuSleePy tests"""

import pytest

from accusleepy.brain_state_set import BrainState, BrainStateSet
from accusleepy.fileio import EMGFilter


@pytest.fixture
def sample_brain_state_set():
    """Default 3-state brain state set"""
    states = [
        BrainState(name="REM", digit=0, is_scored=True, frequency=0.1),
        BrainState(name="Wake", digit=1, is_scored=True, frequency=0.35),
        BrainState(name="NREM", digit=2, is_scored=True, frequency=0.55),
    ]
    return BrainStateSet(states, undefined_label=-1)


@pytest.fixture
def sample_emg_filter():
    """Default EMG filter parameters"""
    return EMGFilter(order=8, bp_lower=20, bp_upper=50)
