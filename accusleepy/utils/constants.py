import numpy as np

from accusleepy.utils.misc import BrainState, BrainStateMapper

# change these
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
IMAGE_HEIGHT = 279 + EMG_COPIES  # TODO determine based on img size?
MIXTURE_WEIGHTS = np.array([0.1, 0.35, 0.55])  # rem, wake, nrem
