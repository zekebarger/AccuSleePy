"""File dialog helpers"""

import os

from PySide6.QtWidgets import QFileDialog, QWidget


def select_existing_file(parent: QWidget, title: str, file_filter: str) -> str | None:
    """Show dialog to select an existing file.

    :param parent: parent widget
    :param title: dialog window title
    :param file_filter: file type filter (e.g., "*.csv")
    :return: normalized path or None if cancelled
    """
    dialog = QFileDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setViewMode(QFileDialog.ViewMode.Detail)
    dialog.setNameFilter(file_filter)

    if dialog.exec():
        return os.path.normpath(dialog.selectedFiles()[0])
    return None


def select_save_location(parent: QWidget, caption: str, file_filter: str) -> str | None:
    """Show dialog to choose save location.

    :param parent: parent widget
    :param caption: dialog window caption
    :param file_filter: file type filter (e.g., "*.csv")
    :return: normalized path or None if cancelled
    """
    filename, _ = QFileDialog.getSaveFileName(
        parent, caption=caption, filter=file_filter
    )
    if filename:
        return os.path.normpath(filename)
    return None
