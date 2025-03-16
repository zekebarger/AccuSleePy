from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PySide6.QtWidgets import *

import numpy as np


# https://stackoverflow.com/questions/67637912/resizing-matplotlib-chart-with-qt5-python


class MplWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())  # layout="tight"))

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        self.canvas.axes = None
        # self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

        # lower plot uses these
        self.eeg_line = None
        self.emg_line = None
        self.top_marker = list()
        self.bottom_marker = list()
        self.rectangles = list()

    def setup_lower_plots(
        self,
        sampling_rate,
        epoch_length,
        epochs_to_show,
        brain_state_mapper,
        label_display_options,
    ):
        # set plot spacing
        gs1 = GridSpec(3, 1, hspace=0)
        gs2 = GridSpec(3, 1, hspace=0.5)
        # create axes
        axes = list()
        axes.append(self.canvas.figure.add_subplot(gs1[0]))
        axes.append(self.canvas.figure.add_subplot(gs1[1]))
        axes.append(self.canvas.figure.add_subplot(gs2[2]))
        # set axis properties
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_yticks([])
        axes[0].set_ylabel("EEG")
        axes[1].set_ylabel("EMG")
        axes[0].set_xlim((0, sampling_rate * epoch_length * epochs_to_show))
        axes[1].set_xlim((0, sampling_rate * epoch_length * epochs_to_show))
        axes[0].set_ylim((-1, 1))
        axes[1].set_ylim((-1, 1))
        axes[2].set_xlim([0, epochs_to_show])
        axes[2].set_ylim(
            [np.min(label_display_options), np.max(label_display_options) + 1]
        )
        axes[2].set_xticks([])
        axes[2].set_yticks([b + 0.5 for b in label_display_options])
        axes[2].set_yticklabels([b.name for b in brain_state_mapper.brain_states])

        # plot markers for selected epoch
        # self.top_marker = [None, None, None]
        marker_dx = sampling_rate * epoch_length
        marker_dy = 0.25
        self.top_marker.append(axes[0].plot([0, 0], [1 - marker_dy, 1], "r")[0])
        self.top_marker.append(axes[0].plot([0, marker_dx], [1, 1], "r")[0])
        self.top_marker.append(
            axes[0].plot([marker_dx, marker_dx], [1 - marker_dy, 1], "r")[0]
        )
        self.bottom_marker.append(axes[0].plot([0, 0], [-1 + marker_dy, -1], "r")[0])
        self.bottom_marker.append(axes[0].plot([0, marker_dx], [-1, -1], "r")[0])
        self.bottom_marker.append(
            axes[0].plot([marker_dx, marker_dx], [-1 + marker_dy, -1], "r")[0]
        )

        # plot placeholders
        self.eeg_line = axes[0].plot(
            np.zeros(int(epochs_to_show * sampling_rate * epoch_length)), "k"
        )[0]
        self.emg_line = axes[1].plot(
            np.zeros(int(epochs_to_show * sampling_rate * epoch_length)), "k"
        )[0]
        for i in range(epochs_to_show):
            self.rectangles.append(
                axes[2].add_patch(Rectangle((i, 0), 1, 1, color=[1, 1, 1, 1]))
            )

        self.canvas.axes = axes
