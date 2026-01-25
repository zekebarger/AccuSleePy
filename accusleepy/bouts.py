import re
from dataclasses import dataclass
from operator import attrgetter

import numpy as np


@dataclass
class Bout:
    """Stores information about a brain state bout"""

    length: int  # length, in number of epochs
    start_index: int  # index where bout starts
    end_index: int  # index where bout ends (non-inclusive)
    surrounding_state: int  # brain state on both sides of the bout


def find_last_adjacent_bout(sorted_bouts: list[Bout], bout_index: int) -> int:
    """Find index of last consecutive same-length bout

     When running the post-processing step that enforces a minimum duration
     for brain state bouts, there is a special case when bouts below the
     duration threshold occur consecutively. This function performs a
     recursive search for the index of a bout at the end of such a sequence.
     When initially called, bout_index will always be 0. If, for example, the
     first three bouts in the list are consecutive, the function will return 2.

    :param sorted_bouts: list of brain state bouts, sorted by start time
    :param bout_index: index of the bout in question
    :return: index of the last consecutive same-length bout
    """
    # if we're at the end of the bout list, stop
    if bout_index == len(sorted_bouts) - 1:
        return bout_index

    # if there is an adjacent bout
    if sorted_bouts[bout_index].end_index == sorted_bouts[bout_index + 1].start_index:
        # look for more adjacent bouts using that one as a starting point
        return find_last_adjacent_bout(sorted_bouts, bout_index + 1)
    else:
        return bout_index


def find_short_bouts(
    labels: np.ndarray, min_epochs: int, brain_states: set[int]
) -> list[Bout]:
    """Locate all brain state bouts below a minimum length

    :param labels: brain state labels (digits in the 0-9 range)
    :param min_epochs: minimum number of epochs in a bout
    :param brain_states: set of brain states in the labels
    :return: list of Bout objects
    """
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
                f"(?<={other_state}){state}{{1,{min_epochs - 1}}}(?={other_state})"
            )
            matches = re.finditer(expression, label_string)
            spans = [match.span() for match in matches]

            for span in spans:
                bouts.append(
                    Bout(
                        length=span[1] - span[0],
                        start_index=span[0],
                        end_index=span[1],
                        surrounding_state=other_state,
                    )
                )
    return bouts


def enforce_min_bout_length(
    labels: np.ndarray, epoch_length: int | float, min_bout_length: int | float
) -> np.ndarray:
    """Ensure brain state bouts meet the min length requirement

    As a post-processing step for sleep scoring, we can require that any
    bout (continuous period) of a brain state have a minimum duration.
    This function sets any bout shorter than the minimum duration to the
    surrounding brain state (if the states on the left and right sides
    are the same). In the case where there are consecutive short bouts,
    it either creates a transition at the midpoint or removes all short
    bouts, depending on whether the number is even or odd. For example:
    ...AAABABAAA...  -> ...AAAAAAAAA...
    ...AAABABABBB... -> ...AAAAABBBBB...

    :param labels: brain state labels (digits in the 0-9 range)
    :param epoch_length: epoch length, in seconds
    :param min_bout_length: minimum bout length, in seconds
    :return: updated brain state labels
    """
    # if the recording is very short or the minimum bout length
    # is one epoch long, don't change anything
    if labels.size < 3 or epoch_length == min_bout_length:
        return labels

    # get minimum number of epochs in a bout
    min_epochs = int(np.ceil(min_bout_length / epoch_length))
    # get set of states in the labels
    brain_states = set(labels.tolist())

    while True:
        bouts = find_short_bouts(labels, min_epochs, brain_states)
        if len(bouts) == 0:
            break

        # only keep the shortest bouts
        min_length_in_list = np.min([bout.length for bout in bouts])
        bouts = [i for i in bouts if i.length == min_length_in_list]
        # sort by start index
        sorted_bouts = sorted(bouts, key=attrgetter("start_index"))

        while len(sorted_bouts) > 0:
            # get row index of latest adjacent bout (of same length)
            last_adjacent_bout_index = find_last_adjacent_bout(sorted_bouts, 0)
            # if there's an even number of adjacent bouts
            if (last_adjacent_bout_index + 1) % 2 == 0:
                midpoint = sorted_bouts[
                    round((last_adjacent_bout_index + 1) / 2)
                ].start_index
                labels[sorted_bouts[0].start_index : midpoint] = sorted_bouts[
                    0
                ].surrounding_state
                labels[midpoint : sorted_bouts[last_adjacent_bout_index].end_index] = (
                    sorted_bouts[last_adjacent_bout_index].surrounding_state
                )
            else:
                labels[
                    sorted_bouts[0].start_index : sorted_bouts[
                        last_adjacent_bout_index
                    ].end_index
                ] = sorted_bouts[0].surrounding_state

            # delete the bouts we just fixed
            if last_adjacent_bout_index == len(sorted_bouts) - 1:
                sorted_bouts = []
            else:
                sorted_bouts = sorted_bouts[(last_adjacent_bout_index + 1) :]

    return labels
