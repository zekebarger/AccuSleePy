from accusleepy.utils.misc import BrainState, BrainStateMapper

# # change these as needed # #
# Recommended to set the digits in the order they appear on the keyboard, 1234567890
# There is no need to set an "undefined" state - this is -1 by default (see below)
# It's not crucial that the typical brain state frequency is accurate, but it helps
BRAIN_STATES = [
    BrainState("REM", 1, True, 0.1),
    BrainState("Wake", 2, True, 0.35),
    BrainState("NREM", 3, True, 0.55),
]


# # probably don't change these unless you really need to # #
UNDEFINED_LABEL = -1  # can't be the same as a digit in BRAIN_STATES, must be an integer
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


# # really don't change these # #
BRAIN_STATE_MAPPER = BrainStateMapper(BRAIN_STATES, UNDEFINED_LABEL)
EMG_COPIES = 9
MIN_WINDOW_LEN = 5
DOWNSAMPLING_START_FREQ = 20
UPPER_FREQ = 50
DEFAULT_MODEL_TYPE = "default"
REAL_TIME_MODEL_TYPE = "real-time"
KEY_TO_MODEL_TYPE = {0: DEFAULT_MODEL_TYPE, 1: REAL_TIME_MODEL_TYPE}
MODEL_TYPE_TO_KEY = {DEFAULT_MODEL_TYPE: 0, REAL_TIME_MODEL_TYPE: 1}
