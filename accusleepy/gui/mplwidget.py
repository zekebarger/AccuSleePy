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
        self.canvas = FigureCanvas(Figure())

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        self.canvas.axes = None
        # self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

        # lower plot uses this
        self.rectangles = list()

    def setup_lower_plots(self, sampling_rate, epochs_to_show, brain_state_mapper):
        gs1 = GridSpec(3, 1, hspace=0)
        gs2 = GridSpec(3, 1, hspace=0.5)
        axes = list()
        axes.append(self.canvas.figure.add_subplot(gs1[0]))
        axes.append(self.canvas.figure.add_subplot(gs1[1]))
        axes.append(self.canvas.figure.add_subplot(gs2[2]))
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_yticks([])
        axes[0].set_ylabel("EEG")
        axes[1].set_ylabel("EMG")
        axes[0].set_xlim((0, sampling_rate * epochs_to_show))
        axes[1].set_xlim((0, sampling_rate * epochs_to_show))

        axes[2].set_xlim([0, epochs_to_show])
        axes[2].set_ylim([0, brain_state_mapper.n_classes])
        axes[2].set_xticks([])
        axes[2].set_yticks(np.arange(brain_state_mapper.n_classes) + 0.5)
        axes[2].set_yticklabels([b.name for b in brain_state_mapper.brain_states])

        for i in range(epochs_to_show):
            self.rectangles.append(
                axes[2].add_patch(Rectangle((i, 0), 1, 1, color=[1, 1, 1, 1]))
            )

        self.canvas.axes = axes
