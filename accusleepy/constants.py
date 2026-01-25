import numpy as np

# probably don't change these unless you really need to
UNDEFINED_LABEL = -1  # can't be the same as a brain state's digit, must be an integer
# calibration file columns
MIXTURE_MEAN_COL = "mixture_mean"
MIXTURE_SD_COL = "mixture_sd"
# recording file columns
EEG_COL = "eeg"
EMG_COL = "emg"
# label file columns
BRAIN_STATE_COL = "brain_state"
CONFIDENCE_SCORE_COL = "confidence_score"
# max number of messages to store in main window message box
MESSAGE_BOX_MAX_DEPTH = 200
# clip mixture z-scores above and below this level
# in the matlab implementation, 4.5 was used
ABS_MAX_Z_SCORE = 3.5
# upper frequency limit when generating EEG spectrograms
SPECTROGRAM_UPPER_FREQ = 64
# minimum number of epochs per brain state needed to create
# a calibration file
MIN_EPOCHS_PER_STATE = 3


# very unlikely you will want to change values from here onwards
# config file location
CONFIG_FILE = "config.json"
# number of times to include the EMG power in a training image
EMG_COPIES = 9
# minimum spectrogram window length, in seconds
MIN_WINDOW_LEN = 5
# frequency above which to downsample EEG spectrograms
DOWNSAMPLING_START_FREQ = 20
# highest EEG frequency used as model input
UPPER_FREQ = 50
# height in pixels of each training image
IMAGE_HEIGHT = (
    len(np.arange(0, DOWNSAMPLING_START_FREQ, 1 / MIN_WINDOW_LEN))
    + len(np.arange(DOWNSAMPLING_START_FREQ, UPPER_FREQ, 2 / MIN_WINDOW_LEN))
    + EMG_COPIES
)

# classification model types
DEFAULT_MODEL_TYPE = "default"  # current epoch is centered
REAL_TIME_MODEL_TYPE = "real-time"  # current epoch on the right
# valid filetypes
RECORDING_FILE_TYPES = [".parquet", ".csv"]
LABEL_FILE_TYPE = ".csv"
CALIBRATION_FILE_TYPE = ".csv"
MODEL_FILE_TYPE = ".pth"
# annotation file columns
FILENAME_COL = "filename"
LABEL_COL = "label"
# recording list file header:
RECORDING_LIST_NAME = "recording_list"
RECORDING_LIST_FILE_TYPE = ".json"
# filename used to store info about training image datasets
ANNOTATIONS_FILENAME = "annotations.csv"
# filename for annotation file for the calibration set
CALIBRATION_ANNOTATION_FILENAME = "calibration_set.csv"

# config file keys
# ui setting keys
DEFAULT_EPOCH_LENGTH_KEY = "default_epoch_length"
DEFAULT_CONFIDENCE_SETTING_KEY = "save_confidence_setting"
DEFAULT_MIN_BOUT_LENGTH_KEY = "default_min_bout_length"
DEFAULT_OVERWRITE_KEY = "default_overwrite_setting"
# EMG filter parameters key
EMG_FILTER_KEY = "emg_filter"
# model training hyperparameters key
HYPERPARAMETERS_KEY = "hyperparameters"
EPOCHS_TO_SHOW_KEY = "epochs_to_show"
AUTOSCROLL_KEY = "autoscroll_state"

# default values
# default UI settings
DEFAULT_MIN_BOUT_LENGTH = 5.0
DEFAULT_CONFIDENCE_SETTING = True
DEFAULT_OVERWRITE_SETTING = False
# default EMG filter parameters (order, bandpass frequencies)
DEFAULT_EMG_FILTER_ORDER = 8
DEFAULT_EMG_BP_LOWER = 20
DEFAULT_EMG_BP_UPPER = 50
# default hyperparameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MOMENTUM = 0.9
DEFAULT_TRAINING_EPOCHS = 6
# default manual scoring settings
DEFAULT_EPOCHS_TO_SHOW = 5
DEFAULT_AUTOSCROLL_STATE = False
