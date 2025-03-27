# Widget that displays a matplotlib FigureCanvas
from collections.abc import Callable

import matplotlib.ticker as mticker
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PySide6.QtWidgets import *

from accusleepy.brain_state_set import BrainStateSet

# upper limit of spectrogram y-axis, in Hz
SPEC_UPPER_F = 30
# interval of spectrogram y-axis ticks, in Hz
SPEC_Y_TICK_INTERVAL = 10

# margins around subplots in the figure
SUBPLOT_TOP_MARGIN = 0.98
SUBPLOT_BOTTOM_MARGIN = 0.02
SUBPLOT_LEFT_MARGIN = 0.07
SUBPLOT_RIGHT_MARGIN = 0.95

# maximum number of x-axis ticks to show on the lower plot
MAX_LOWER_X_TICK_N = 7


class MplWidget(QWidget):
    """Widget that displays a matplotlib FigureCanvas"""

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        # set up the canvas and store a reference to its axes
        self.canvas = FigureCanvas(Figure())
        self.canvas.axes = None

        # set the widget layout and remove the margins
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vertical_layout)

        # given during the setup process
        self.epoch_length = None

        # upper plot references
        self.upper_marker = None
        self.label_img_ref = None
        self.spec_ref = None
        self.roi = None
        self.editing_patch = None
        self.roi_patch = None

        # lower plot references
        self.eeg_line = None
        self.emg_line = None
        self.top_marker = None
        self.bottom_marker = None

    def setup_upper_figure(
        self,
        n_epochs: int,
        label_img: np.array,
        spec: np.array,
        f: np.array,
        emg: np.array,
        epochs_to_show: int,
        label_display_options: np.array,
        brain_state_set: BrainStateSet,
        roi_function: Callable,
    ):
        """Initialize upper FigureCanvas for the manual scoring GUI

        :param n_epochs: number of epochs in the recording
        :param label_img: brain state labels, as an image
        :param spec: EEG spectrogram image
        :param f: EEG spectrogram frequency axis
        :param emg: EMG RMS per epoch
        :param epochs_to_show: number of epochs to show in the lower plot
        :param label_display_options: valid brain state y-axis locations
        :param brain_state_set: set of brain states options
        :param roi_function: callback for ROI selection
        """
        # references to parts of the epoch marker
        self.upper_marker = list()

        # subplot layout
        height_ratios = [8, 2, 12, 13]
        gs1 = GridSpec(4, 1, hspace=0, height_ratios=height_ratios)
        gs2 = GridSpec(4, 1, hspace=0.4, height_ratios=height_ratios)
        axes = list()
        axes.append(self.canvas.figure.add_subplot(gs1[0]))
        axes.append(self.canvas.figure.add_subplot(gs1[1]))
        axes.append(self.canvas.figure.add_subplot(gs1[2]))
        axes.append(self.canvas.figure.add_subplot(gs2[3]))

        # subplots have different axes limits
        for i in [0, 1, 3]:
            axes[i].set_xlim((-0.5, n_epochs + 0.5))
        axes[2].set_xlim(0, n_epochs)

        # brain state subplot
        axes[0].set_ylim(
            [-0.5, np.max(label_display_options) - np.min(label_display_options) + 0.5]
        )
        axes[0].set_xticks([])
        axes[0].set_yticks(
            label_display_options - np.min(label_display_options),
        )
        axes[0].set_yticklabels([b.name for b in brain_state_set.brain_states])
        ax2 = axes[0].secondary_yaxis("right")
        ax2.set_yticks(
            label_display_options - np.min(label_display_options),
        )
        ax2.set_yticklabels([b.digit for b in brain_state_set.brain_states])
        self.label_img_ref = axes[0].imshow(
            label_img, aspect="auto", origin="lower", interpolation="None"
        )
        # add patch to dim the display when creating an ROI
        self.editing_patch = axes[0].add_patch(
            Rectangle(
                xy=(-0.5, -0.5),
                width=n_epochs + 1,
                height=np.max(label_display_options)
                - np.min(label_display_options)
                + 1,
                color="white",
                edgecolor=None,
                alpha=0.4,
                visible=False,
            )
        )
        # add the ROI selection widget, but disable it until it's needed
        self.roi = RectangleSelector(
            ax=axes[0],
            onselect=roi_function,
            interactive=False,
            button=MouseButton(1),
        )
        self.roi.set_active(False)
        # keep a reference to the ROI patch so we can change its color later
        # index 0 is the "editing_patch" created earlier
        self.roi_patch = [c for c in axes[0].get_children() if type(c) == Rectangle][1]

        # epoch marker subplot
        axes[1].set_ylim((0, 1))
        axes[1].axis("off")
        self.upper_marker.append(
            axes[1].plot([-0.5, epochs_to_show - 0.5], [0.5, 0.5], "r")[0]
        )
        self.upper_marker.append(axes[1].plot([0], [0.5], "rD")[0])

        # EEG spectrogram subplot
        # select subset of frequencies to show
        f = f[f <= SPEC_UPPER_F]
        spec = spec[0 : len(f), :]
        axes[2].set_ylabel("EEG", rotation="horizontal", ha="right")
        axes[2].set_yticks(
            np.linspace(
                0,
                len(f),
                1 + int(SPEC_UPPER_F / SPEC_Y_TICK_INTERVAL),
            ),
        )
        axes[2].set_yticklabels(
            np.arange(0, SPEC_UPPER_F + SPEC_Y_TICK_INTERVAL, SPEC_Y_TICK_INTERVAL)
        )
        axes[2].tick_params(axis="both", which="major", labelsize=8)
        axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(self.time_formatter))
        self.spec_ref = axes[2].imshow(
            spec,
            vmin=np.percentile(spec, 2),
            vmax=np.percentile(spec, 98),
            aspect="auto",
            origin="lower",
            interpolation="None",
            extent=(
                0,
                n_epochs,
                -0.5,
                len(f) + 0.5,
            ),
        )

        # EMG subplot
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_ylabel("EMG", rotation="horizontal", ha="right")
        axes[3].plot(
            emg,
            "k",
            linewidth=0.5,
        )

        self.canvas.figure.subplots_adjust(
            left=SUBPLOT_LEFT_MARGIN,
            right=SUBPLOT_RIGHT_MARGIN,
            top=SUBPLOT_TOP_MARGIN,
            bottom=SUBPLOT_BOTTOM_MARGIN,
        )

        self.canvas.axes = axes

    def setup_lower_figure(
        self,
        label_img: np.array,
        sampling_rate: int | float,
        epochs_to_show: int,
        brain_state_set: BrainStateSet,
        label_display_options: np.array,
    ):
        """Initialize lower FigureCanvas for the manual scoring GUI

        :param label_img: brain state labels, as an image
        :param sampling_rate: EEG/EMG sampling rate, in Hz
        :param epochs_to_show: number of epochs to show in the lower plot
        :param brain_state_set: set of brain states options
        :param label_display_options: valid brain state y-axis locations
        """
        # number of samples in one epoch
        samples_per_epoch = sampling_rate * self.epoch_length
        # number of EEG/EMG samples to plot
        samples_shown = samples_per_epoch * epochs_to_show

        # references to parts of the epoch markers
        self.top_marker = list()
        self.bottom_marker = list()
        # epoch marker display parameters
        marker_dy = 0.25
        marker_y_offset_top = 0.02
        marker_y_offset_bottom = 0.01

        # subplot layout
        gs1 = GridSpec(3, 1, hspace=0)
        gs2 = GridSpec(3, 1, hspace=0.5)
        axes = list()
        axes.append(self.canvas.figure.add_subplot(gs1[0]))
        axes.append(self.canvas.figure.add_subplot(gs1[1]))
        axes.append(self.canvas.figure.add_subplot(gs2[2]))

        # EEG subplot
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlim((0, samples_shown))
        axes[0].set_ylim((-1, 1))
        axes[0].set_ylabel("EEG", rotation="horizontal", ha="right")
        self.eeg_line = axes[0].plot(
            np.zeros(int(samples_shown)),
            "k",
            linewidth=0.5,
        )[0]
        # top epoch marker
        marker_x = [
            [0, 0],
            [0, samples_per_epoch],
            [samples_per_epoch, samples_per_epoch],
        ]
        marker_y = np.array(
            [
                [1 - marker_dy, 1],
                [1, 1],
                [1 - marker_dy, 1],
            ]
        )
        for x, y in zip(marker_x, marker_y):
            self.top_marker.append(axes[0].plot(x, y - marker_y_offset_top, "r")[0])

        # EMG subplot
        axes[1].set_xticks(
            resample_x_ticks(
                np.arange(
                    0,
                    samples_shown,
                    samples_per_epoch,
                )
            )
        )
        axes[1].tick_params(axis="x", which="major", labelsize=8)
        axes[1].set_yticks([])
        axes[1].set_xlim((0, samples_shown))
        axes[1].set_ylim((-1, 1))
        axes[1].set_ylabel("EMG", rotation="horizontal", ha="right")
        self.emg_line = axes[1].plot(
            np.zeros(int(samples_shown)),
            "k",
            linewidth=0.5,
        )[0]

        for x, y in zip(marker_x, marker_y):
            self.bottom_marker.append(
                axes[1].plot(x, -1 * (y - marker_y_offset_bottom), "r")[0]
            )

        # brain state subplot
        axes[2].set_xticks([])
        axes[2].set_yticks(
            label_display_options - np.min(label_display_options),
        )
        axes[2].set_yticklabels([b.name for b in brain_state_set.brain_states])
        ax2 = axes[2].secondary_yaxis("right")
        ax2.set_yticks(
            label_display_options - np.min(label_display_options),
        )
        ax2.set_yticklabels([b.digit for b in brain_state_set.brain_states])
        axes[2].set_xlim((-0.5, epochs_to_show - 0.5))
        axes[2].set_ylim(
            [-0.5, np.max(label_display_options) - np.min(label_display_options) + 0.5]
        )
        self.label_img_ref = axes[2].imshow(
            label_img[:, 0:epochs_to_show, :],
            aspect="auto",
            origin="lower",
            interpolation="None",
        )

        self.canvas.figure.subplots_adjust(
            left=SUBPLOT_LEFT_MARGIN,
            right=SUBPLOT_RIGHT_MARGIN,
            top=SUBPLOT_TOP_MARGIN,
            bottom=SUBPLOT_BOTTOM_MARGIN,
        )

        self.canvas.axes = axes

    def time_formatter(self, x, pos):
        x = x * self.epoch_length
        return "{:02d}:{:02d}:{:05.2f}".format(
            int(x // 3600), int(x // 60) % 60, (x % 60)
        )


def resample_x_ticks(x_ticks: np.array) -> np.array:
    """Choose a subset of x_ticks to display

    The x-axis can get crowded if there are too many timestamps shown.
    This function resamples the x-axis ticks by a factor of either
    MAX_LOWER_X_TICK_N or MAX_LOWER_X_TICK_N - 2, whichever is closer
    to being a factor of the number of ticks.

    :param x_ticks: full set of x_ticks
    :return: smaller subset of x_ticks
    """
    # add one since the tick at the rightmost edge isn't shown
    n_ticks = len(x_ticks) + 1
    if n_ticks < MAX_LOWER_X_TICK_N:
        return x_ticks
    elif n_ticks % MAX_LOWER_X_TICK_N < n_ticks % (MAX_LOWER_X_TICK_N - 2):
        return x_ticks[:: int(n_ticks / MAX_LOWER_X_TICK_N)]
    else:
        return x_ticks[:: int(n_ticks / (MAX_LOWER_X_TICK_N - 2))]
