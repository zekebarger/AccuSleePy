"""Recording list manager"""

from dataclasses import dataclass

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from accusleepy.fileio import Recording, load_recording_list, save_recording_list


@dataclass
class RecordingListItem(Recording):
    """A Recording with an associated QListWidget item for the GUI"""

    widget: QListWidgetItem = None


class RecordingListManager(QObject):
    """Manages the list of recordings and the associated QListWidget"""

    def __init__(self, list_widget: QListWidget, parent: QObject | None = None):
        super().__init__(parent)
        self._widget = list_widget

        # Create initial empty recording (there is always at least one)
        first_recording = RecordingListItem(
            widget=QListWidgetItem("Recording 1", self._widget),
        )
        self._recordings: list[RecordingListItem] = [first_recording]
        self._widget.addItem(first_recording.widget)
        self._widget.setCurrentRow(0)

    @property
    def current(self) -> RecordingListItem:
        """The currently selected recording"""
        return self._recordings[self._widget.currentRow()]

    def add(self, sampling_rate: int | float) -> RecordingListItem:
        """Add a new recording to the list

        :param sampling_rate: sampling rate for the new recording
        :return: the newly created Recording
        """
        new_name = max(r.name for r in self._recordings) + 1

        # Create recording with widget
        recording = RecordingListItem(
            name=new_name,
            sampling_rate=sampling_rate,
            widget=QListWidgetItem(f"Recording {new_name}", self._widget),
        )
        self._recordings.append(recording)

        # Update widget
        self._widget.addItem(recording.widget)
        self._widget.setCurrentRow(len(self._recordings) - 1)

        return recording

    def remove_current(self) -> str:
        """Remove the currently selected recording

        :return: message describing what was removed/reset
        """
        if len(self._recordings) > 1:
            # Remove from list and widget
            index = self._widget.currentRow()
            recording_name = self._recordings[index].name
            self._widget.takeItem(index)
            del self._recordings[index]
            return f"deleted Recording {recording_name}"
        else:
            # Reset the single recording to defaults
            recording_name = self._recordings[0].name
            self._recordings[0] = RecordingListItem(widget=self._recordings[0].widget)
            self._recordings[0].widget.setText(f"Recording {self._recordings[0].name}")
            return f"cleared Recording {recording_name}"

    def export_to_file(self, filename: str) -> None:
        """Save the recording list to a file

        :param filename: path to which the list will be exported
        """
        save_recording_list(filename=filename, recordings=self._recordings)

    def import_from_file(self, filename: str) -> None:
        """Load a recording list from a file, replacing current list

        :param filename: path to load from
        """
        # Block signals while rebuilding the list to avoid triggering
        # selection callbacks with invalid state
        self._widget.blockSignals(True)
        try:
            self._widget.clear()

            # Load recordings and create widgets
            self._recordings = [
                RecordingListItem(
                    **r.__dict__,
                    widget=QListWidgetItem(f"Recording {r.name}", self._widget),
                )
                for r in load_recording_list(filename)
            ]
            for recording in self._recordings:
                self._widget.addItem(recording.widget)
        finally:
            self._widget.blockSignals(False)

        # Select first recording (this will trigger the selection callback)
        self._widget.setCurrentRow(0)

    def __iter__(self):
        return iter(self._recordings)

    def __len__(self):
        return len(self._recordings)

    def __getitem__(self, index: int) -> RecordingListItem:
        return self._recordings[index]
