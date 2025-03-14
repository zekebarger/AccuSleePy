# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'window1.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QMainWindow, QSizePolicy, QWidget)

from mplwidget import MplWidget

class Ui_Window1(object):
    def setupUi(self, Window1):
        if not Window1.objectName():
            Window1.setObjectName(u"Window1")
        Window1.resize(475, 263)
        self.centralwidget = QWidget(Window1)
        self.centralwidget.setObjectName(u"centralwidget")
        self.mplwidget1 = MplWidget(self.centralwidget)
        self.mplwidget1.setObjectName(u"mplwidget1")
        self.mplwidget1.setGeometry(QRect(70, 40, 301, 151))
        Window1.setCentralWidget(self.centralwidget)

        self.retranslateUi(Window1)

        QMetaObject.connectSlotsByName(Window1)
    # setupUi

    def retranslateUi(self, Window1):
        Window1.setWindowTitle(QCoreApplication.translate("Window1", u"MainWindow", None))
    # retranslateUi

