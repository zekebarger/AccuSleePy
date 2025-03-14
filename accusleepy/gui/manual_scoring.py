# import sys
import random
import matplotlib

matplotlib.use("QtAgg")
from PySide6 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from accusleepy.utils.fileio import load_labels, load_recording
from accusleepy.utils.misc import Recording

KEY_MAP = {
    QtCore.Qt.Key.Key_Backspace: "backspace",
    QtCore.Qt.Key.Key_Tab: "tab",
    # Add mappings for other keys you need to support
    QtCore.Qt.Key.Key_Return: "enter",
    QtCore.Qt.Key.Key_Enter: "enter",
    QtCore.Qt.Key.Key_Escape: "esc",
    QtCore.Qt.Key.Key_Space: "space",
    QtCore.Qt.Key.Key_End: "end",
    QtCore.Qt.Key.Key_Home: "home",
    QtCore.Qt.Key.Key_Left: "left",
    QtCore.Qt.Key.Key_Up: "up",
    QtCore.Qt.Key.Key_Right: "right",
    QtCore.Qt.Key.Key_Down: "down",
    QtCore.Qt.Key.Key_Delete: "delete",
}

# THESE SHOULD BE ENTERED
sampling_rate = 512
epoch_length = 2.5

# SHOULD BE SET BY USER
epochs_to_show = 5


class WindowContents(QtWidgets.QWidget):

    def __init__(self, eeg, emg, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # time index
        self.t = 100

        self.canvas = RawSignalPlot(self)
        # self.setCentralWidget(self.canvas)

        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [random.randint(0, 10) for i in range(n_data)]

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self._plot_ref = None
        self.update_plot()

        self.button = QtWidgets.QPushButton("Plot")
        self.button.clicked.connect(self.update_plot)

        # set the layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(
            self.canvas,
            0,
            0,
            2,
            1,
        )
        layout.addWidget(self.button, 1, 0, 2, 1)
        self.setLayout(layout)

        self.key_label = QtWidgets.QLabel("Last Key Pressed: None", self.canvas)

    def update_plot(self, direction="right"):
        print("updating")
        if direction == "left":
            self.ydata = [random.randint(0, 10)] + self.ydata[:-1]
        else:
            self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        # Note: we no longer need to clear the axis.
        if self._plot_ref is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_refs = self.canvas.ax1.plot(self.xdata, self.ydata, "r")
            self._plot_ref = plot_refs[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref.set_ydata(self.ydata)

        # Trigger the canvas to update and redraw.
        self.canvas.draw()


class RawSignalPlot(FigureCanvas):

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        self.ax1 = ax1
        self.ax2 = ax2
        super().__init__(fig)

        # functions to scroll should live HERE


class MainWindow(QtWidgets.QMainWindow):

    def __init__(
        self,
        eeg,
        emg,
        labels,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setGeometry(100, 100, 1000, 800)

        window_contents = WindowContents(
            eeg,
            emg,
            labels,
        )
        # widget.setLayout(layout)
        self.setCentralWidget(window_contents)

        keypress_up = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
        # this isn't working.
        keypress_up.activated.connect(
            self.centralWidget().update_plot(direction="right")
        )

        self.show()

    # # use QShortcut instead??
    # def keyPressEvent(self, event):
    #     if isinstance(event, QtGui.QKeyEvent):
    #         key = None
    #         if event.key() in KEY_MAP:
    #             key = KEY_MAP[event.key()]
    #         else:
    #             key = event.text()
    #         self.centralWidget().key_label.setText(f"Last Key Pressed: {key}")
    #         self.keypress_handler(key)

    # def keypress_handler(self, key):
    #     if key == "left":
    #         self.centralWidget().update_plot(direction=key)
    #     elif key == "right":
    #         self.centralWidget().update_plot(direction=key)


if __name__ == "__main__":
    import sys

    # just for testing
    sampling_rate = 512
    epoch_length = 2.5
    mouse_1_recordings = [
        Recording(
            recording_file=f"/Users/zeke/Documents/data/Mouse01/Day{i}/recording.parquet",
            label_file=f"/Users/zeke/Documents/data/Mouse01/Day{i}/labels.csv",
            sampling_rate=sampling_rate,
        )
        for i in range(1, 6)
    ]
    eeg, emg = load_recording(mouse_1_recordings[0].recording_file)
    labels = load_labels(mouse_1_recordings[0].label_file)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(eeg, emg, labels)
    window.setGeometry(400, 200, 800, 600)
    window.show()
    sys.exit(app.exec())
