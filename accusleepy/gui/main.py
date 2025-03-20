# AccuSleePy main window

import sys

from PySide6 import QtCore, QtGui, QtWidgets
from Window0 import Ui_Window0


class MainWindow(QtWidgets.QMainWindow):
    """AccuSleePy main window"""

    def __init__(self):
        super(MainWindow, self).__init__()

        # initialize the UI
        self.ui = Ui_Window0()
        self.ui.setupUi(self)
        self.setWindowTitle("AccuSleePy")

        keypress_quit = QtGui.QShortcut(
            QtGui.QKeySequence(
                QtCore.QKeyCombination(QtCore.Qt.Modifier.CTRL, QtCore.Qt.Key.Key_W)
            ),
            self,
        )
        keypress_quit.activated.connect(self.close)

        self.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
