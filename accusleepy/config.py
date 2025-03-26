import json
import os

from accusleepy.misc import BRAIN_STATES_KEY, BrainState, BrainStateMapper

# # change these as needed # #
# Recommended to set the digits in the order they appear on the keyboard, 1234567890
# There is no need to set an "undefined" state - this is -1 by default (see below)
# It's not crucial that the typical brain state frequency is accurate, but it helps
# todo move this text


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
EMG_COPIES = 9
MIN_WINDOW_LEN = 5
DOWNSAMPLING_START_FREQ = 20
UPPER_FREQ = 50
DEFAULT_MODEL_TYPE = "default"
REAL_TIME_MODEL_TYPE = "real-time"
KEY_TO_MODEL_TYPE = {0: DEFAULT_MODEL_TYPE, 1: REAL_TIME_MODEL_TYPE}
MODEL_TYPE_TO_KEY = {DEFAULT_MODEL_TYPE: 0, REAL_TIME_MODEL_TYPE: 1}
RECORDING_FILE_TYPES = [".parquet", ".csv"]
LABEL_FILE_TYPE = ".csv"
CALIBRATION_FILE_TYPE = ".csv"
MODEL_FILE_TYPE = ".pth"
CONFIG_FILE = "config.json"


def load_config() -> BrainStateMapper:
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE), "r"
    ) as f:
        data = json.load(f)
    return BrainStateMapper(
        [BrainState(**b) for b in data[BRAIN_STATES_KEY]], UNDEFINED_LABEL
    )


def save_config(brain_state_mapper: BrainStateMapper) -> None:
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE), "w"
    ) as f:
        json.dump(brain_state_mapper.output_dict(), f, indent=4)
