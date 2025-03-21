# AccuSleePy main window

import sys

from PySide6 import QtCore, QtGui, QtWidgets
from Window0 import Ui_Window0

from accusleepy.utils.misc import Recording


MESSAGE_BOX_MAX_DEPTH = 20


class MainWindow(QtWidgets.QMainWindow):
    """AccuSleePy main window"""

    def __init__(self):
        super(MainWindow, self).__init__()

        # initialize the UI
        self.ui = Ui_Window0()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy")

        # set up the list of recordings
        # create empty recording
        first_recording = Recording(
            widget=QtWidgets.QListWidgetItem(
                "Recording 1", self.ui.recording_list_widget
            ),
        )
        # show it in the list widget
        self.ui.recording_list_widget.addItem(first_recording.widget)
        self.ui.recording_list_widget.setCurrentRow(0)

        # index of currently selected recording in the list
        self.recording_index = 0
        # list of recordings the user has added
        self.recordings = [first_recording]

        # messages to display
        self.messages = []

        # user input
        # keyboard shortcuts
        keypress_quit = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(QtCore.Qt.Modifier.CTRL, QtCore.Qt.Key.Key_W)
            ),
            self,
        )
        keypress_quit.activated.connect(self.close)

        # button presses
        self.ui.add_button.clicked.connect(self.add_recording)
        self.ui.remove_button.clicked.connect(self.remove_recording)
        self.ui.recording_list_widget.currentRowChanged.connect(self.select_recording)

        self.show()

    def show_message(self, message: str) -> None:
        """Display a new message to the user"""
        self.messages.append(message)
        if len(self.messages) > MESSAGE_BOX_MAX_DEPTH:
            del self.messages[0]
        self.ui.message_area.setText("\n".join(self.messages))
        # scroll to the bottom
        scrollbar = self.ui.message_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def select_recording(self, list_index) -> None:
        """Callback for when a recording is selected"""
        self.show_message(f"selected Recording {self.recordings[list_index].name}")

    def add_recording(self) -> None:
        """Add new recording to the list"""
        # find name to use for the new recording
        new_name = max([r.name for r in self.recordings]) + 1

        # add new recording to list
        # TODO: insert sampling rate
        self.recordings.append(
            Recording(
                name=new_name,
                widget=QtWidgets.QListWidgetItem(
                    f"Recording {new_name}", self.ui.recording_list_widget
                ),
            )
        )

        # display new list
        self.ui.recording_list_widget.addItem(self.recordings[-1].widget)
        self.ui.recording_list_widget.setCurrentRow(len(self.recordings) - 1)
        self.show_message(f"added Recording {new_name}")

    def remove_recording(self) -> None:
        """Delete selected recording from the list"""
        if len(self.recordings) > 1:
            current_list_index = self.ui.recording_list_widget.currentRow()
            _ = self.ui.recording_list_widget.takeItem(current_list_index)
            self.show_message(
                f"deleted Recording {self.recordings[current_list_index].name}"
            )
            del self.recordings[current_list_index]


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
