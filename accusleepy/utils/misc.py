from dataclasses import dataclass

import numpy as np

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


class BrainStateMapper:
    def __init__(self, brain_states):
        self.brain_states = brain_states

        self.digit_to_class = dict()
        self.class_to_digit = dict()
        i = 0
        for brain_state in self.brain_states:
            if brain_state.is_scored:
                self.digit_to_class[brain_state.digit] = i
                self.class_to_digit[i] = brain_state.digit
                i += 1
            else:
                self.digit_to_class[brain_state.digit] = None

        self.n_classes = i

    def convert_digit_to_class(self, digits):
        return np.array([self.digit_to_class[i] for i in digits])

    def convert_class_to_digit(self, classes):
        return np.array([self.class_to_digit[i] for i in classes])


@dataclass
class Recording:
    recording_file: str
    label_file: str
    sampling_rate: int | float
