from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import QListWidgetItem

BRAIN_STATES_KEY = "brain_states"

# It's convenient to have the brain state labels start at 1 rather than 0.
# However, they need to be converted to the 0-n range for training and inference.
# So, we have to have a distinction between "brain states" (as represented in
# label files, keystrokes) and "classes" (AccuSleep's internal representation).
# This is confusing and I apologize.


@dataclass
class BrainState:
    name: str  # friendly name
    digit: int  # number 0-9 - used as keyboard shortcut
    is_scored: bool  # whether a classification model should score this state
    frequency: int | float  # typical relative frequency, between 0 and 1


class BrainStateMapper:
    def __init__(self, brain_states, undefined_label):
        self.brain_states = brain_states

        self.digit_to_class = {undefined_label: None}
        self.class_to_digit = dict()
        self.mixture_weights = list()
        i = 0
        for brain_state in self.brain_states:
            if brain_state.digit == undefined_label:
                raise Exception(
                    f"Digit for {brain_state.name} matches 'undefined' label"
                )
            if brain_state.is_scored:
                self.digit_to_class[brain_state.digit] = i
                self.class_to_digit[i] = brain_state.digit
                self.mixture_weights.append(brain_state.frequency)
                i += 1
            else:
                self.digit_to_class[brain_state.digit] = None

        self.n_classes = i

        self.mixture_weights = np.array(self.mixture_weights)
        if np.sum(self.mixture_weights) != 1:
            raise Exception(
                f"Typical frequencies for scored brain states must sum to 1"
            )

    def convert_digit_to_class(self, digits):
        return np.array([self.digit_to_class[i] for i in digits])

    def convert_class_to_digit(self, classes):
        return np.array([self.class_to_digit[i] for i in classes])

    def output_dict(self) -> dict:
        return {
            BRAIN_STATES_KEY: [
                {
                    "name": b.name,
                    "digit": b.digit,
                    "is_scored": b.is_scored,
                    "frequency": b.frequency,
                }
                for b in self.brain_states
            ]
        }


@dataclass
class Recording:
    name: int = 1  # name to show in the GUI
    recording_file: str = ""  # path to recording file
    label_file: str = ""  # path to label file
    calibration_file: str = ""  # path to calibration file
    sampling_rate: int | float = 0.0  # sampling rate, in Hz
    widget: QListWidgetItem = None  # reference to widget shown in the GUI
