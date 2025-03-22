# AccuSleePy main window

import sys

from PySide6 import QtCore, QtGui, QtWidgets
from Window0 import Ui_Window0

from accusleepy.utils.misc import Recording


MESSAGE_BOX_MAX_DEPTH = 50


class MainWindow(QtWidgets.QMainWindow):
    """AccuSleePy main window"""

    def __init__(self):
        super(MainWindow, self).__init__()

        # initialize the UI
        self.ui = Ui_Window0()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy")

        self.epoch_length = 0

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
        self.ui.sampling_rate_input.valueChanged.connect(self.update_sampling_rate)
        self.ui.epoch_length_input.valueChanged.connect(self.update_epoch_length)
        self.ui.recording_file_button.clicked.connect(self.select_recording_file)
        self.ui.label_file_button.clicked.connect(self.select_label_file)

        self.show()

    def select_label_file(self) -> None:
        """User can select a label file for this recording"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select label file")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("(*.csv)")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            self.recordings[self.recording_index].label_file = filename
            self.ui.label_file_label.setText(filename)

    def select_recording_file(self) -> None:
        """User can select a recording file"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select recording file")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        file_dialog.setNameFilter("(*.parquet *.csv)")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            filename = selected_files[0]
            self.recordings[self.recording_index].recording_file = filename
            self.ui.recording_file_label.setText(filename)

    def show_recording_info(self) -> None:
        self.ui.sampling_rate_input.setValue(
            self.recordings[self.recording_index].sampling_rate
        )
        self.ui.recording_file_label.setText(
            self.recordings[self.recording_index].recording_file
        )

    def update_epoch_length(self, new_value) -> None:
        self.epoch_length = new_value

    def update_sampling_rate(self, new_value) -> None:
        self.recordings[self.recording_index].sampling_rate = new_value

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
        # get index of this recording
        self.recording_index = list_index
        # display information about this recording
        self.show_recording_info()
        self.ui.selected_recording_groupbox.setTitle(
            (
                "Data / actions for the selected recording "
                f"(Recording {self.recordings[list_index].name}) "
                "from this subject)"
            )
        )

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
