import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

app = QApplication(sys.argv)

window = QMainWindow()
window.setWindowTitle("first mainwindow app")

button = QPushButton()
button.setText("press me")

window.setCentralWidget(button)

window.show()
app.exec()