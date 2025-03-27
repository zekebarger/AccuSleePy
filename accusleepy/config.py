# # probably don't change these unless you really need to # #
UNDEFINED_LABEL = -1  # can't be the same as a brain state's digit, must be an integer
# calibration file columns
MIXTURE_MEAN_COL = "mixture_mean"
MIXTURE_SD_COL = "mixture_sd"
# recording file columns
EEG_COL = "eeg"
EMG_COL = "emg"
# label file columns
BRAIN_STATE_COL = "brain_state"


# # really don't change these # #
CONFIG_FILE = "config.json"
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
# annotation file columns
FILENAME_COL = "filename"
LABEL_COL = "label"
