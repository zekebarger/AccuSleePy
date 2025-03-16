import numpy as np

from accusleepy.utils.misc import BrainState, BrainStateMapper

# change these
# recommended to set the digits in the order they appear on the keyboard, 1234567890
BRAIN_STATES = [
    BrainState("REM", 1, True),
    BrainState("Wake", 2, True),
    BrainState("NREM", 3, True),
]

BRAIN_STATE_MAPPER = BrainStateMapper(BRAIN_STATES)

EPOCHS_PER_IMG = 9


# probably don't change these
# annotation file columns
FILENAME_COL = "filename"
LABEL_COL = "label"
# calibration file columns
MIXTURE_MEAN_COL = "mixture_mean"
MIXTURE_SD_COL = "mixture_sd"
# recording file columns
EEG_COL = "eeg"
EMG_COL = "emg"
BRAIN_STATE_COL = "brain_state"


# really don't change these
EMG_COPIES = 9
MIN_WINDOW_LEN = 5
DOWNSAMPLING_START_FREQ = 20
UPPER_FREQ = 50
IMAGE_HEIGHT = (
    len(np.arange(0, DOWNSAMPLING_START_FREQ, 1 / MIN_WINDOW_LEN))
    + len(np.arange(DOWNSAMPLING_START_FREQ, UPPER_FREQ, 2 / MIN_WINDOW_LEN))
    + EMG_COPIES
)
MIXTURE_WEIGHTS = np.array([0.1, 0.35, 0.55])  # rem, wake, nrem
