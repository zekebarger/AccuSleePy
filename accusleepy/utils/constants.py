import numpy as np

N_CLASSES = 3

# annotation file columns
FILENAME_COL = "filename"
LABEL_COL = "label"
# calibration file columns
MIXTURE_MEAN_COL = "mixture_mean"
MIXTURE_SD_COL = "mixture_sd"

EMG_COPIES = 9
EPOCHS_PER_IMG = 9

MIXTURE_WEIGHTS = np.array([0.1, 0.35, 0.55])  # rem, wake, nrem
