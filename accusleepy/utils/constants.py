import numpy as np

from accusleepy.utils.misc import BrainState, BrainStateMapper

BRAIN_STATES = [
    BrainState("REM", 1, True),
    BrainState("Wake", 2, True),
    BrainState("NREM", 3, True),
]

BRAIN_STATE_MAPPER = BrainStateMapper(BRAIN_STATES)

# annotation file columns
FILENAME_COL = "filename"
LABEL_COL = "label"
# calibration file columns
MIXTURE_MEAN_COL = "mixture_mean"
MIXTURE_SD_COL = "mixture_sd"
# recording file columns
EEG_COL = "eeg"
EMG_COL = "emg"

EMG_COPIES = 9
EPOCHS_PER_IMG = 9

MIXTURE_WEIGHTS = np.array([0.1, 0.35, 0.55])  # rem, wake, nrem
