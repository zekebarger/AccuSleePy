import re
from dataclasses import dataclass
from operator import attrgetter

import numpy as np
from PySide6.QtWidgets import QListWidgetItem

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
    def __init__(self, brain_states, undefined_label):
        self.brain_states = brain_states

        self.digit_to_class = {undefined_label: None}
        self.class_to_digit = dict()
        i = 0
        for brain_state in self.brain_states:
            if brain_state.digit == undefined_label:
                raise Exception(
                    f"Digit for {brain_state.name} matches 'undefined' label"
                )
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
    name: int = 1  # name to show in the GUI
    recording_file: str = ""  # path to recording file
    label_file: str = ""  # path to label file
    sampling_rate: int | float = 0.0  # sampling rate, in Hz
    widget: QListWidgetItem = None  # reference to widget shown in the GUI


@dataclass
class Bout:
    length: int
    start_index: int
    end_index: int
    # state: int
    surrounding_state: int


# get row index of latest adjacent bout (of same length) recursively
def find_last_adjacent_bout(sorted_bouts, bout_index):
    # if we're at the end of the bout list, stop
    if bout_index == len(sorted_bouts) - 1:
        return bout_index

    # if there is an adjacent bout
    if sorted_bouts[bout_index].end_index == sorted_bouts[bout_index + 1].start_index:
        # look for more adjacent bouts using that one as a starting point
        return find_last_adjacent_bout(sorted_bouts, bout_index + 1)
    else:
        return bout_index


def enforce_min_bout_length(
    labels: np.array, epoch_length: int | float, min_bout_length: int | float
) -> np.array:
    # all entries in labels must be digits in the 0-9 range

    # if recording is very short, don't change anything
    if labels.size < 3:
        return labels

    # get minimum number of epochs in a bout
    min_epochs = int(np.ceil(min_bout_length / epoch_length))
    # get set of states in the labels
    brain_states = set(labels.tolist())

    while True:  # so true
        # convert labels to a string for regex search
        # There is probably a regex that can find all patterns like ab+a
        # without consuming each "a" but I haven't found it :(
        label_string = "".join(labels.astype(str))

        bouts = list()

        for state in brain_states:
            for other_state in brain_states:
                if state == other_state:
                    continue
                # get start and end indices of each bout
                expression = (
                    f"(?<={other_state}){state}{{1,{min_epochs-1}}}(?={other_state})"
                )
                matches = re.finditer(expression, label_string)
                spans = [match.span() for match in matches]

                # if some bouts were found
                for span in spans:
                    bouts.append(
                        Bout(
                            length=span[1] - span[0],
                            start_index=span[0],
                            end_index=span[1],
                            # state=state,
                            surrounding_state=other_state,
                        )
                    )

        if len(bouts) == 0:
            break

        # only keep the shortest bouts
        min_length_in_list = np.min([bout.length for bout in bouts])
        bouts = [i for i in bouts if i.length == min_length_in_list]
        # sort by start index
        sorted_bouts = sorted(bouts, key=attrgetter("start_index"))

        # Work through the list. If several bouts of the same length are
        # adjacent, either eliminate them all or create a transition halfway
        # through if there are an even or odd number of short bouts,
        # respectively.
        while len(sorted_bouts) > 0:
            # get row index of latest adjacent bout (of same length)
            last_adjacent_bout_index = find_last_adjacent_bout(sorted_bouts, 0)
            # if there's an even number of adjacent bouts
            if (last_adjacent_bout_index + 1) % 2 == 0:
                midpoint = sorted_bouts[
                    int((last_adjacent_bout_index + 1) / 2)
                ].start_index
                labels[sorted_bouts[0].start_index : midpoint] = sorted_bouts[
                    0
                ].surrounding_state
                labels[midpoint : sorted_bouts[last_adjacent_bout_index].end_index] = (
                    sorted_bouts[last_adjacent_bout_index].surrounding_state
                )
            else:
                labels[
                    sorted_bouts[0]
                    .start_index : sorted_bouts[last_adjacent_bout_index]
                    .end_index
                ] = sorted_bouts[0].surrounding_state

            # delete the bouts we just fixed
            if last_adjacent_bout_index == len(sorted_bouts) - 1:
                sorted_bouts = []
            else:
                sorted_bouts = sorted_bouts[(last_adjacent_bout_index + 1) :]

    return labels
