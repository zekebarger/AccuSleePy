from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PySide6.QtWidgets import *

import numpy as np


# https://stackoverflow.com/questions/67637912/resizing-matplotlib-chart-with-qt5-python

SPEC_UPPER_F = 30
SPEC_YTICK_INTERVAL = 10


class MplWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())  # layout="tight"))

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas.axes = None
        self.setLayout(vertical_layout)

        # upper plot uses these
        self.upper_marker = list()
        self.label_img_ref = None

        # lower plot uses these
        self.eeg_line = None
        self.emg_line = None
        self.top_marker = list()
        self.bottom_marker = list()
        self.rectangles = list()

    def setup_upper_plots(
        self,
        n_epochs,
        label_img,
        spec,
        f,
        emg,
        epochs_to_show,
        label_display_options,
        brain_state_mapper,
    ):
        height_ratios = [8, 2, 12, 13]
        gs1 = GridSpec(4, 1, hspace=0, height_ratios=height_ratios)
        gs2 = GridSpec(4, 1, hspace=0.4, height_ratios=height_ratios)
        axes = list()
        axes.append(self.canvas.figure.add_subplot(gs1[0]))
        axes.append(self.canvas.figure.add_subplot(gs1[1]))
        axes.append(self.canvas.figure.add_subplot(gs1[2]))
        axes.append(self.canvas.figure.add_subplot(gs2[3]))
        self.canvas.figure.subplots_adjust(top=0.98, bottom=0.02, right=0.98)

        for i in range(3):
            axes[i].set_xlim((-0.5, n_epochs + 0.5))
        axes[3].set_xlim((0, n_epochs))

        # brain states
        axes[0].set_xticks([])
        axes[0].set_yticks(
            label_display_options - np.min(label_display_options),
        )
        axes[0].set_yticklabels([b.name for b in brain_state_mapper.brain_states])
        axes[0].set_ylim(
            [-0.5, np.max(label_display_options) - np.min(label_display_options) + 0.5]
        )
        self.label_img_ref = axes[0].imshow(
            label_img, aspect="auto", origin="lower", interpolation="None"
        )

        # epoch marker
        # axes[1].axis("off") # use this eventually
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_ylim((0, 1))
        # line
        self.upper_marker.append(
            axes[1].plot([-0.5, epochs_to_show - 0.5], [0.5, 0.5], "r")[0]
        )
        # marker
        self.upper_marker.append(axes[1].plot([0], [0.5], "rD")[0])

        # spectrogram
        f = f[f <= SPEC_UPPER_F]
        spec = spec[0 : len(f), :]
        axes[2].set_ylabel("EEG", rotation="horizontal", ha="right")
        axes[2].set_yticks(
            np.linspace(
                0,
                len(f),
                1 + int(SPEC_UPPER_F / SPEC_YTICK_INTERVAL),
            ),
        )
        # todo need to fix the outermost bins being centered on 0, ...
        axes[2].set_yticklabels(
            [
                f"{i} hz"
                for i in np.arange(
                    0, SPEC_UPPER_F + SPEC_YTICK_INTERVAL, SPEC_YTICK_INTERVAL
                )
            ]
        )
        axes[2].imshow(
            spec,
            vmin=np.percentile(spec, 2),
            vmax=np.percentile(spec, 98),
            aspect="auto",
            origin="lower",
            interpolation="None",
        )

        # emg
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_ylabel("EMG", rotation="horizontal", ha="right")
        axes[3].plot(emg, "k")

        self.canvas.axes = axes

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
        self.canvas.figure.subplots_adjust(top=0.98, bottom=0.02, right=0.98)

        marker_dx = sampling_rate * epoch_length
        marker_dy = 0.25

        # set axis properties
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlim((0, sampling_rate * epoch_length * epochs_to_show))
        axes[0].set_ylim((-1, 1))
        axes[0].set_ylabel("EEG", rotation="horizontal", ha="right")
        self.eeg_line = axes[0].plot(
            np.zeros(int(epochs_to_show * sampling_rate * epoch_length)), "k"
        )[0]
        # plot markers for selected epoch
        self.top_marker.append(axes[0].plot([0, 0], [1 - marker_dy, 1], "r")[0])
        self.top_marker.append(axes[0].plot([0, marker_dx], [1, 1], "r")[0])
        self.top_marker.append(
            axes[0].plot([marker_dx, marker_dx], [1 - marker_dy, 1], "r")[0]
        )

        axes[1].set_yticks([])
        axes[1].set_ylabel("EMG", rotation="horizontal", ha="right")
        axes[1].set_xlim((0, sampling_rate * epoch_length * epochs_to_show))
        axes[1].set_ylim((-1, 1))
        self.emg_line = axes[1].plot(
            np.zeros(int(epochs_to_show * sampling_rate * epoch_length)), "k"
        )[0]
        self.bottom_marker.append(axes[1].plot([0, 0], [-1 + marker_dy, -1], "r")[0])
        self.bottom_marker.append(axes[1].plot([0, marker_dx], [-1, -1], "r")[0])
        self.bottom_marker.append(
            axes[1].plot([marker_dx, marker_dx], [-1 + marker_dy, -1], "r")[0]
        )

        axes[2].set_xlim([0, epochs_to_show])
        axes[2].set_ylim(
            [np.min(label_display_options), np.max(label_display_options) + 1]
        )
        axes[2].set_xticks([])
        axes[2].set_yticks([b + 0.5 for b in label_display_options])
        axes[2].set_yticklabels([b.name for b in brain_state_mapper.brain_states])
        for i in range(epochs_to_show):
            self.rectangles.append(
                axes[2].add_patch(Rectangle((i, 0), 1, 1, color=[1, 1, 1, 1]))
            )

        self.canvas.axes = axes
