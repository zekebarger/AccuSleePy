"""Colors for the primary window.

These are applied to the primary window only, so the manual scoring window
keeps whatever the current style provides.

The idea behind the colors is that light surfaces are the ones you can act
on (buttons, text fields) and darker surfaces are empty space. Two things
about the Fusion style shape how that is achieved.

First, Fusion paints the tab pages, the group boxes, and the button faces
all from the Button role. The Window role only sets the outline around a
button, which is the window color darkened by 140. Because the pages and
the buttons share one role, a palette on its own cannot lift a button more
than about 8 levels of lightness above the page behind it.

So the pages are repainted with a stylesheet instead, which leaves the
Button role free to keep the buttons light. That brings the button faces to
a lightness of 245 against a page of 220. The stylesheet names the
containers by object name and never mentions QPushButton, which matters:
any stylesheet rule on a button makes Qt recompute its size hints from the
stylesheet and drop the metrics the style would have given it.

Base stays white, which keeps text fields the brightest surfaces of all.

Note that widgets with a hardcoded stylesheet ignore all of this. In the
primary window that includes the recording list, the file path labels, the
Settings tab description text, and the logo, all of which set their own
background or text color in primary_window.ui.
"""

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette

# background color of the tab pages and group boxes
PAGE_COLOR = "#dcdcdc"

# tab pages and group boxes that get repainted
_PAGE_WIDGETS = (
    "scoring_tab",
    "settings_tab",
    "classification_tab",
    "model_training_tab",
)
_GROUP_BOXES = (
    "primary_defaults_groupbox",
    "manual_defaults_groupbox",
    "recordinglistgroupbox",
    "selected_recording_groupbox",
    "messagesgroupbox",
)

# window sets the button outline, button sets the button faces
_WINDOW = "#c8c8c8"
_BUTTON = "#f4f4f4"
_TEXT = "#000000"
_BASE = "#ffffff"
_HIGHLIGHT = "#0050c8"
# borders, for the widgets that do use these roles
_MID = "#4a4a4a"
_DARK = "#222222"


@dataclass
class PaletteSpec:
    """A palette, plus the background for the pages behind it"""

    palette: QPalette
    page: str


def page_stylesheet(color: str) -> str:
    """Build a stylesheet that repaints the tab pages and group boxes

    Only container widgets are named, so buttons keep the size hints the
    style gives them.

    :param color: background color to apply
    :return: the stylesheet
    """
    rules = [
        f"QWidget#{name} {{ background-color: {color}; }}" for name in _PAGE_WIDGETS
    ]
    rules += [
        f"QGroupBox#{name} {{ background-color: {color}; }}" for name in _GROUP_BOXES
    ]
    return "\n".join(rules)


def primary_window_palette() -> PaletteSpec:
    """Build the colors for the primary window

    :return: the palette and the page background that goes with it
    """
    palette = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window, _WINDOW),
        (QPalette.ColorRole.WindowText, _TEXT),
        (QPalette.ColorRole.Base, _BASE),
        (QPalette.ColorRole.AlternateBase, "#eef0f2"),
        (QPalette.ColorRole.Text, _TEXT),
        (QPalette.ColorRole.Button, _BUTTON),
        (QPalette.ColorRole.ButtonText, _TEXT),
        (QPalette.ColorRole.Highlight, _HIGHLIGHT),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.Mid, _MID),
        (QPalette.ColorRole.Dark, _DARK),
        (QPalette.ColorRole.ToolTipBase, _BASE),
        (QPalette.ColorRole.ToolTipText, _TEXT),
    ]:
        palette.setColor(role, QColor(color))

    # keep disabled widgets legible but clearly inactive
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, QColor(_MID))

    return PaletteSpec(palette=palette, page=PAGE_COLOR)
