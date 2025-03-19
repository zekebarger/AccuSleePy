import matplotlib.ticker as mticker
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PySide6.QtWidgets import *

from accusleepy.utils.constants import MAX_LOWER_XTICK_N

# icons from Arkinasi, https://www.flaticon.com/authors/arkinasi
# and kendis lasman, https://www.flaticon.com/packs/ui-79

SPEC_UPPER_F = 30
SPEC_YTICK_INTERVAL = 10

LEFT_MARGIN = 0.07
RIGHT_MARGIN = 0.95


def resample_x_ticks(x_ticks):
    """Choose a subset of x_ticks to display

    :param x_ticks: full set of x_ticks
    :return: smaller subset of x_ticks
    """
    n_ticks = len(x_ticks) + 1  # add imaginary one at the end
    if n_ticks < MAX_LOWER_XTICK_N:
        return x_ticks
    if n_ticks % MAX_LOWER_XTICK_N < n_ticks % (MAX_LOWER_XTICK_N - 2):
        x_ticks = x_ticks[:: int(n_ticks / MAX_LOWER_XTICK_N)]
    else:
        x_ticks = x_ticks[:: int(n_ticks / (MAX_LOWER_XTICK_N - 2))]
    return x_ticks


class MplWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas.axes = None
        self.setLayout(vertical_layout)

        # upper plot uses these
        self.upper_marker = None
        self.label_img_ref = None
        self.spec_ref = None
        self.roi = None
        self.roi_patch = None

        # lower plot uses these
        self.eeg_line = None
        self.emg_line = None
        self.top_marker = None
        self.bottom_marker = None

        self.epoch_length = None

    def setup_upper_plots(
        self,
        n_epochs,
        label_img,
        spec,
        f,
        emg,
        epoch_length,
        epochs_to_show,
        label_display_options,
        brain_state_mapper,
        roi_function,
    ):
        self.epoch_length = epoch_length
        self.upper_marker = list()
        height_ratios = [8, 2, 12, 13]
        gs1 = GridSpec(4, 1, hspace=0, height_ratios=height_ratios)
        gs2 = GridSpec(4, 1, hspace=0.4, height_ratios=height_ratios)
        axes = list()
        axes.append(self.canvas.figure.add_subplot(gs1[0]))
        axes.append(self.canvas.figure.add_subplot(gs1[1]))
        axes.append(self.canvas.figure.add_subplot(gs1[2]))
        axes.append(self.canvas.figure.add_subplot(gs2[3]))
        self.canvas.figure.subplots_adjust(top=0.98, bottom=0.02, right=0.98)

        for i in range(4):
            axes[i].set_xlim((-0.5, n_epochs + 0.5))

        # brain states
        axes[0].set_xticks([])
        axes[0].set_yticks(
            label_display_options - np.min(label_display_options),
        )
        axes[0].set_yticklabels([b.name for b in brain_state_mapper.brain_states])
        ax2 = axes[0].secondary_yaxis("right")
        ax2.set_yticks(
            label_display_options - np.min(label_display_options),
        )
        ax2.set_yticklabels([b.digit for b in brain_state_mapper.brain_states])

        axes[0].set_ylim(
            [-0.5, np.max(label_display_options) - np.min(label_display_options) + 0.5]
        )
        self.label_img_ref = axes[0].imshow(
            label_img, aspect="auto", origin="lower", interpolation="None"
        )
        self.roi = RectangleSelector(
            ax=axes[0],
            onselect=roi_function,
            interactive=False,
            button=MouseButton(1),
        )
        self.roi.set_active(False)
        # since there are no other rectangles except the background,
        # we can keep a reference to the ROI patch so we can change its color
        self.roi_patch = [c for c in axes[0].get_children() if type(c) == Rectangle][0]

        # epoch marker
        axes[1].axis("off")
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
        axes[2].set_yticklabels(
            [
                i  # f"{i} hz"
                for i in np.arange(
                    0, SPEC_UPPER_F + SPEC_YTICK_INTERVAL, SPEC_YTICK_INTERVAL
                )
            ]
        )
        self.spec_ref = axes[2].imshow(
            spec,
            vmin=np.percentile(spec, 2),
            vmax=np.percentile(spec, 98),
            aspect="auto",
            origin="lower",
            interpolation="None",
        )
        axes[2].tick_params(axis="both", which="major", labelsize=8)
        axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(self.fmtsec))

        # emg
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_ylabel("EMG", rotation="horizontal", ha="right")
        axes[3].plot(
            emg,
            "k",
            linewidth=0.5,
        )

        self.canvas.figure.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN)

        self.canvas.axes = axes

    def setup_lower_plots(
        self,
        label_img,
        sampling_rate,
        epoch_length,
        epochs_to_show,
        brain_state_mapper,
        label_display_options,
    ):
        self.top_marker = list()
        self.bottom_marker = list()

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
            np.zeros(int(epochs_to_show * sampling_rate * epoch_length)),
            "k",
            linewidth=0.5,
        )[0]
        # plot markers for selected epoch
        self.top_marker.append(axes[0].plot([0, 0], [1 - marker_dy, 1], "r")[0])
        self.top_marker.append(axes[0].plot([0, marker_dx], [1, 1], "r")[0])
        self.top_marker.append(
            axes[0].plot([marker_dx, marker_dx], [1 - marker_dy, 1], "r")[0]
        )

        x_ticks = resample_x_ticks(
            np.arange(
                0,
                sampling_rate * epoch_length * epochs_to_show,
                sampling_rate * epoch_length,
            )
        )
        axes[1].set_xticks(x_ticks)
        axes[1].set_yticks([])
        axes[1].tick_params(axis="x", which="major", labelsize=8)
        axes[1].set_ylabel("EMG", rotation="horizontal", ha="right")
        axes[1].set_xlim((0, sampling_rate * epoch_length * epochs_to_show))
        axes[1].set_ylim((-1, 1))
        self.emg_line = axes[1].plot(
            np.zeros(int(epochs_to_show * sampling_rate * epoch_length)),
            "k",
            linewidth=0.5,
        )[0]
        self.bottom_marker.append(axes[1].plot([0, 0], [-1 + marker_dy, -1], "r")[0])
        self.bottom_marker.append(axes[1].plot([0, marker_dx], [-1, -1], "r")[0])
        self.bottom_marker.append(
            axes[1].plot([marker_dx, marker_dx], [-1 + marker_dy, -1], "r")[0]
        )

        # brain states
        axes[2].set_xlim((-0.5, epochs_to_show - 0.5))
        axes[2].set_xticks([])
        axes[2].set_yticks(
            label_display_options - np.min(label_display_options),
        )
        axes[2].set_yticklabels([b.name for b in brain_state_mapper.brain_states])
        axes[2].set_ylim(
            [-0.5, np.max(label_display_options) - np.min(label_display_options) + 0.5]
        )
        self.label_img_ref = axes[2].imshow(
            label_img[:, 0:epochs_to_show, :],
            aspect="auto",
            origin="lower",
            interpolation="None",
        )
        ax2 = axes[2].secondary_yaxis("right")
        ax2.set_yticks(
            label_display_options - np.min(label_display_options),
        )
        ax2.set_yticklabels([b.digit for b in brain_state_mapper.brain_states])

        self.canvas.figure.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN)

        self.canvas.axes = axes

    def fmtsec(self, x, pos):
        x = (x + 0.5) * self.epoch_length
        return "{:02d}:{:02d}:{:05.2f}".format(int(x // 3600), int(x // 60), (x % 60))
