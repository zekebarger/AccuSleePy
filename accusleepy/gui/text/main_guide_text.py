from accusleepy.constants import (
    BRAIN_STATE_COL,
    CALIBRATION_FILE_TYPE,
    EEG_COL,
    EMG_COL,
    LABEL_FILE_TYPE,
    MODEL_FILE_TYPE,
    RECORDING_FILE_TYPES,
    UNDEFINED_LABEL,
)

MAIN_GUIDE_TEXT = f"""
Section 0: Definitions
Section 1: Overview of the GUI
Section 2: AccuSleePy file types
Section 3: Manually assigning brain state labels
Section 4: Automatically assigning brain state labels

-----------------------------------------------------------------------
Section 0: Definitions
-----------------------------------------------------------------------
Recording: a table containing one channel of EEG data and one channel
    of EMG data collected at a constant sampling rate.
Epoch: the temporal resolution of brain state scoring. If, for example,
    the epoch length is 5 seconds, then a brain state label will be
    assigned to each 5-second segment of a recording.
Bout: a contiguous set of epochs with the same brain state.

-----------------------------------------------------------------------
Section 1: Overview of the primary interface
-----------------------------------------------------------------------
This interface allows a user to assign brain state labels to 1-channel
electroencephalogram (EEG) and electromyogram (EMG) data.
The overall workflow looks like this:
1. Enter the epoch length for all recordings. This determines the time
    resolution of the brain state labels.
2. For each of your recordings, add to the recording list, enter its
    sampling rate, load the EEG/EMG data, and determine where to save
    the brain state labels (or load the labels if they already exist)
At this point, you can score the recordings manually.
3. For each recording, create a calibration file using a small amount
    of labeled data, or choose a calibration file created using
    another recording from the same subject and under the same recording
    conditions (i.e., equipment)
4. Choose a trained classification model file with a matching epoch size
5. Score all recordings automatically using the classifier

By default, there are three brain state options: REM, wake, and NREM.
If you want to change this configuration, click the "Settings" tab.
Note that if you change the configuration, you might be unable to load
existing labels and calibration data, and you may need to train a new
classification model.

Use the "import" and "export" buttons to load or save a list of
recordings. This can be useful if you need to re-score a set of
recordings with a different model, or if you want to keep a record of
the recordings that were used when training your model.

-----------------------------------------------------------------------
Section 2: AccuSleePy file types
-----------------------------------------------------------------------
There are four types of files associated with AccuSleePy.
To select a file in the primary interface, you can either use the
associated button, or drag/drop the file into the empty box adjacent
to the button.
Recording file: a {" or ".join(RECORDING_FILE_TYPES)} file containing one
    column of EEG  data and one column of EMG data.
    The column names must be {EEG_COL} and {EMG_COL}.
Label file: a {LABEL_FILE_TYPE} file with one column titled {BRAIN_STATE_COL}
    with entries that are either the undefined brain state ({UNDEFINED_LABEL})
    or one of the digits in your brain state configuration.
    By default, these are 1-3 where REM = 1, wake = 2, NREM = 3.
Calibration data file: required for automatic labeling. See Section 4
    for details. These have {CALIBRATION_FILE_TYPE} format.
Trained classification model: required for automatic labeling. See
    Section 4 for details. These have {MODEL_FILE_TYPE} format.

-----------------------------------------------------------------------
Section 3: Manually assigning brain state labels
-----------------------------------------------------------------------
1. Select the recording you wish to modify from the recording list, or
    add a new one.
2. Click the 'Select recording file' button to set the location of the
    EEG/EMG data, or drag/drop the recording file into the box next
    to the button.
3. Click the 'Select' label file button (or drag/drop) to choose an
    existing label file, or click the 'create' label file button to
    enter the filename for a new label file.
4. Click 'Score manually' to launch an interactive window for manual
    brain state labeling. Close the window when you are finished.
This interface has many useful keyboard shortcuts, so it's recommended
to consult its user manual.

-----------------------------------------------------------------------
Section 4: Automatically scoring recordings with a classification model
-----------------------------------------------------------------------
Automatic brain state scoring requires the inputs described in
Section 3, as well as calibration data files and a trained classifier.
If you already have all of these files, proceed to Section 4C.
Models trained on the AccuSleep dataset are provided at
https://osf.io/py5eb under /python_format/models/ for epoch lengths of
2.5, 4, 5, and 10 seconds. These models are the "default" type, in that
they use several epochs of data before and after any given epoch when
scoring that epoch. (The other model type, called "real-time", only
uses data from the current epoch and several preceding epochs.)

--- Section 4A: Creating calibration data files ---
Each recording must have a calibration file assigned to it.
This file lets AccuSleep transform features of the EEG and EMG data so
that they are in the same range as the classifier's training data.
You can use the same calibration file for multiple recordings, as long
as they come from the same subject and were collected under the same
recording conditions (i.e., the same recording equipment was used).
To create a calibration data file:
1. Ensure you have a file containing brain state labels. You can create
    this file by following the steps in Section 3, or select an
    existing label file.
2. The label file must contain at least some labels for each sleep
    stage (e.g., REM, wakefulness, and NREM). It is recommended to
    label at least several minutes of each stage, and more labels can
    improve classification accuracy.
3. Click 'Create calibration file'.
4. Enter a filename for the calibration data file.
5. The calibration file will automatically be assigned to the currently
    selected recording.
Note that epoch length can affect the calibration process. If you make
a calibration file for a subject using one epoch length, but want to
score another recording from the same subject with a different epoch
length, it's best to create a new calibration file.

--- Section 4B: Training your own classification model ---
To train a new model on your own data:
1. Add your scored recordings to the recording list. Make sure the
    sampling rate, recording file, and label file are set for each
    recording.
2. Click the "Model training" tab
3. Choose the number of epochs to consider when scoring each epoch.
    This will be the "width" of the training images. For "default"
    type models, this must be an odd number. In general, about 30
    seconds worth of data is enough.
4. Choose whether the images used to train the model should be
    deleted once training is complete. (It's generally best to
    leave this box checked.)
5. Choose whether to create a "default" or "real-time"-type model.
    Note that scoring recordings in the GUI requires a default-type
    model.
6. Select a directory where the training images will be saved. A
    new directory with an automatically generated name will be
    created inside the directory you choose.
7. Click the "Train classification model" button and enter a
    filename for the trained model. Training can take some time.

--- Section 4C: Automatic labeling ---
Instructions for automatic labeling using this GUI are below.
1. Set the epoch length for all recordings.
2. Select the recording file, label file, and calibration file to use
    for each recording. See section 4A for instructions on creating
    calibration files.
3. Click 'Load classification model' to load the trained classification
    model. It's important that the epoch length used when training this
    model is the same as the current epoch length.
4. If you wish to preserve any existing labels in the label file, and
    only overwrite undefined epochs, check the box labeled
    'Only overwrite undefined epochs'.
5. Set the minimum bout length, in seconds. A typical value could be 5.
    Following automatic labeling, any brain state bout shorter than this
    duration will be reassigned to the surrounding stage (if the stages
    on either side of the bout match).
6. Click 'Score all automatically' to score all recordings in the
    recording list. Labels will be saved to the file specified by
    the 'Select or create label file' field of each recording. You can
    click 'Score manually' to visualize the results.
"""
