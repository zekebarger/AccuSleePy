"""Define and organize brain states."""

from dataclasses import dataclass

import numpy as np

BRAIN_STATES_KEY = "brain_states"


@dataclass
class BrainState:
    """Convenience class for a brain state and its attributes"""

    name: str  # friendly name
    digit: int  # number 0-9 - used as keyboard shortcut and in label files
    is_scored: bool  # whether a classification model should score this state
    frequency: int | float  # typical relative frequency, between 0 and 1


class BrainStateSet:
    def __init__(self, brain_states: list[BrainState], undefined_label: int):
        """Initialize set of brain states

        :param brain_states: list of BrainState objects
        :param undefined_label: label for undefined epochs
        """
        self.brain_states = brain_states

        # The user can choose any subset of the digits 0-9 to represent
        # brain states, but not all of them are necessarily intended to be
        # scored by a classifier, and pytorch requires that all input
        # labels are in the 0-n range for training and inference.
        # So, we have to have a distinction between "brain states" (as
        # represented in label files and keyboard inputs) and "classes"
        # (AccuSleep's internal representation).

        # map digits to classes, and vice versa
        self.digit_to_class = {undefined_label: None}
        self.class_to_digit = dict()
        # relative frequencies of each class
        self.mixture_weights = list()

        i = 0
        for brain_state in self.brain_states:
            if brain_state.digit == undefined_label:
                raise ValueError(
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
            raise ValueError(
                "Typical frequencies for scored brain states must sum to 1"
            )

    def convert_digit_to_class(self, digits: np.ndarray) -> np.ndarray:
        """Convert array of digits to their corresponding classes

        :param digits: array of digits
        :return: array of classes
        """
        return np.array([self.digit_to_class[i] for i in digits])

    def convert_class_to_digit(self, classes: np.ndarray) -> np.ndarray:
        """Convert array of classes to their corresponding digits

        :param classes: array of classes
        :return: array of digits
        """
        return np.array([self.class_to_digit[i] for i in classes])

    def to_output_dict(self) -> dict:
        """Return dictionary of brain states"""
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
